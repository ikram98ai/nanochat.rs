//! A nice and efficient mixed AdamW/Muon Combined Optimizer.
//! Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
//!
//! Muon internally runs standard SGD-momentum, and then performs an orthogonalization
//! post-processing step (MuonEq row equilibration, Polar Express, Muon+ renormalization),
//! followed by NorMuon variance reduction and a cautious weight-decay update.
//!
//! Muon should not be used for the embedding layer, the final fully connected layer,
//! or any {0,1}-D parameters; those should all be optimized by AdamW.

use std::collections::HashMap;

use candle_core::{DType, IndexOp, Result, Tensor, Var, D};

/// Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
/// From https://arxiv.org/pdf/2505.16932
const POLAR_EXPRESS_COEFFS: [(f64, f64, f64); 5] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GroupKind {
    AdamW,
    Muon,
}

/// One parameter group. AdamW groups use lr/betas/eps/weight_decay;
/// Muon groups use lr/momentum/ns_steps/beta2/weight_decay.
/// The training loop mutates lr (from initial_lr) and the Muon momentum
/// and weight_decay according to its schedules.
pub struct ParamGroup {
    pub kind: GroupKind,
    pub params: Vec<Var>,
    pub lr: f64,
    pub initial_lr: f64,
    pub weight_decay: f64,
    // AdamW
    pub betas: (f64, f64),
    pub eps: f64,
    // Muon
    pub momentum: f64,
    pub ns_steps: usize,
    pub beta2: f64,
}

impl ParamGroup {
    pub fn adamw(
        params: Vec<Var>,
        lr: f64,
        betas: (f64, f64),
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            kind: GroupKind::AdamW,
            params,
            lr,
            initial_lr: lr,
            weight_decay,
            betas,
            eps,
            momentum: 0.0,
            ns_steps: 0,
            beta2: 0.0,
        }
    }

    pub fn muon(
        params: Vec<Var>,
        lr: f64,
        momentum: f64,
        ns_steps: usize,
        beta2: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            kind: GroupKind::Muon,
            params,
            lr,
            initial_lr: lr,
            weight_decay,
            betas: (0.0, 0.0),
            eps: 0.0,
            momentum,
            ns_steps,
            beta2,
        }
    }
}

struct AdamWState {
    step: u32,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
}

struct MuonState {
    /// Momentum for every individual parameter: (K, M, N)
    momentum_buffer: Tensor,
    /// Factored second momentum, either per-row (K, M, 1) or per-column (K, 1, N)
    second_momentum_buffer: Tensor,
}

/// out = self + weight * (rhs - self), i.e. (1 - weight) * self + weight * rhs
fn lerp(a: &Tensor, b: &Tensor, weight: f64) -> Result<Tensor> {
    (a * (1.0 - weight))? + (b * weight)?
}

/// AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update.
/// Returns the new (param, exp_avg, exp_avg_sq).
fn adamw_step(
    p: &Tensor,
    grad: &Tensor,
    exp_avg: &Tensor,
    exp_avg_sq: &Tensor,
    step: u32,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    wd: f64,
) -> Result<(Tensor, Tensor, Tensor)> {
    // Weight decay (decoupled, applied before the update)
    let p = (p * (1.0 - lr * wd))?;
    // Update running averages
    let exp_avg = lerp(exp_avg, grad, 1.0 - beta1)?;
    let exp_avg_sq = lerp(exp_avg_sq, &grad.sqr()?, 1.0 - beta2)?;
    // Bias corrections (computed in f32 to match the reference 0-D tensors)
    let bias1 = 1.0 - (beta1 as f32).powf(step as f32) as f64;
    let bias2 = 1.0 - (beta2 as f32).powf(step as f32) as f64;
    // Compute update and apply
    let denom = ((&exp_avg_sq / bias2)?.sqrt()? + eps)?;
    let step_size = lr / bias1;
    let p = (p - ((&exp_avg / &denom)? * step_size)?)?;
    Ok((p, exp_avg, exp_avg_sq))
}

/// Muon step on stacked (K, M, N) tensors:
/// momentum -> polar_express -> variance_reduction -> cautious_update.
/// Returns (new_params, new_momentum_buffer, new_second_momentum_buffer).
#[allow(clippy::too_many_arguments)]
fn muon_step(
    stacked_grads: &Tensor,
    stacked_params: &Tensor,
    momentum_buffer: &Tensor,
    second_momentum_buffer: &Tensor,
    momentum: f64,
    lr: f64,
    wd: f64,
    beta2: f64,
    ns_steps: usize,
    red_dim: usize, // actual dim index (1 or 2) of the reduced dimension
) -> Result<(Tensor, Tensor, Tensor)> {
    let (_k, m, n) = stacked_params.dims3()?;

    // Nesterov momentum
    let momentum_buffer = lerp(momentum_buffer, stacked_grads, 1.0 - momentum)?;
    let g = lerp(stacked_grads, &momentum_buffer, momentum)?;

    // MuonEq row equilibration: rescale each row to the mean row norm so the spectrum
    // entering orthogonalization is better conditioned (https://arxiv.org/abs/2603.28254)
    let target = (g.sqr()?.sum_keepdim((1, 2))?.sqrt()? / (m as f64).sqrt())?;
    let row_norm = g.sqr()?.sum_keepdim(2)?.sqrt()?.maximum(1e-6)?;
    let x = g.broadcast_mul(&target.broadcast_div(&row_norm)?)?;

    // Polar Express orthogonalization: replace each update with the nearest orthogonal
    // matrix (compute dtype is f32 on CPU; the bf16 fast path is CUDA-only)
    let norm = x.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let mut x = x.broadcast_div(&((norm * 1.01)? + 1e-6)?)?;
    if m > n {
        // Tall matrix
        for (a, b, c) in POLAR_EXPRESS_COEFFS.iter().take(ns_steps) {
            let xt = x.transpose(D::Minus2, D::Minus1)?.contiguous()?;
            let a_mat = xt.matmul(&x)?;
            let b_mat = ((&a_mat * *b)? + (a_mat.matmul(&a_mat)? * *c)?)?;
            x = ((&x * *a)? + x.matmul(&b_mat)?)?;
        }
    } else {
        // Wide matrix (original math)
        for (a, b, c) in POLAR_EXPRESS_COEFFS.iter().take(ns_steps) {
            let xt = x.transpose(D::Minus2, D::Minus1)?.contiguous()?;
            let a_mat = x.matmul(&xt)?;
            let b_mat = ((&a_mat * *b)? + (a_mat.matmul(&a_mat)? * *c)?)?;
            x = ((&x * *a)? + b_mat.matmul(&x)?)?;
        }
    }
    // Muon+ renormalization: snap the Frobenius norm to sqrt(min(m, n)), the norm of an
    // exactly semi-orthogonal matrix, correcting for under-convergence of the polar
    // iteration (https://arxiv.org/abs/2602.21545)
    let target_norm = (m.min(n) as f64).sqrt();
    let current_norm = x.sqr()?.sum_keepdim((1, 2))?.sqrt()?.maximum(1e-6)?;
    let g = x.broadcast_mul(&(current_norm.recip()? * target_norm)?)?;

    // Variance reduction
    let v_mean = g.sqr()?.mean_keepdim(red_dim)?;
    let red_dim_size = g.dim(red_dim)? as f64;
    let v_norm_sq = (v_mean.sum_keepdim((1, 2))? * red_dim_size)?;
    let v_norm = v_norm_sq.sqrt()?;
    let second_momentum_buffer = lerp(second_momentum_buffer, &v_mean, 1.0 - beta2)?;
    // out-of-place clamp: the buffer itself keeps unclamped values
    let step_size = second_momentum_buffer.maximum(1e-10)?.sqrt()?.recip()?;
    let scaled_sq_sum = ((&v_mean * red_dim_size)?.mul(&step_size.sqr()?))?;
    let v_norm_new = scaled_sq_sum.sum_keepdim((1, 2))?.sqrt()?;
    let final_scale = step_size.broadcast_mul(&v_norm.div(&v_norm_new.maximum(1e-10)?)?)?;
    let g = g.broadcast_mul(&final_scale)?;

    // Cautious weight decay + parameter update
    let mask = g.mul(stacked_params)?.ge(0.0)?.to_dtype(DType::F32)?;
    let update = ((&g * lr)? + ((stacked_params.mul(&mask)?) * (lr * wd))?)?;
    let new_params = (stacked_params - update)?;
    Ok((new_params, momentum_buffer, second_momentum_buffer))
}

/// Combined optimizer: Muon for 2D matrix params, AdamW for others.
pub struct MuonAdamW {
    pub param_groups: Vec<ParamGroup>,
    adamw_state: HashMap<(usize, usize), AdamWState>,
    muon_state: HashMap<usize, MuonState>,
}

impl MuonAdamW {
    pub fn new(param_groups: Vec<ParamGroup>) -> Self {
        Self {
            param_groups,
            adamw_state: HashMap::new(),
            muon_state: HashMap::new(),
        }
    }

    fn step_adamw(
        group_idx: usize,
        group: &ParamGroup,
        adamw_state: &mut HashMap<(usize, usize), AdamWState>,
        get_grad: &dyn Fn(&Var) -> Option<Tensor>,
    ) -> Result<()> {
        for (param_idx, p) in group.params.iter().enumerate() {
            let grad = match get_grad(p) {
                Some(grad) => grad,
                None => continue,
            };
            let state = match adamw_state.entry((group_idx, param_idx)) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => e.insert(AdamWState {
                    step: 0,
                    exp_avg: p.zeros_like()?,
                    exp_avg_sq: p.zeros_like()?,
                }),
            };
            state.step += 1;
            let (new_p, exp_avg, exp_avg_sq) = adamw_step(
                p.as_tensor(),
                &grad,
                &state.exp_avg,
                &state.exp_avg_sq,
                state.step,
                group.lr,
                group.betas.0,
                group.betas.1,
                group.eps,
                group.weight_decay,
            )?;
            p.set(&new_p)?;
            state.exp_avg = exp_avg;
            state.exp_avg_sq = exp_avg_sq;
        }
        Ok(())
    }

    fn step_muon(
        group_idx: usize,
        group: &ParamGroup,
        muon_state: &mut HashMap<usize, MuonState>,
        get_grad: &dyn Fn(&Var) -> Option<Tensor>,
    ) -> Result<()> {
        let params = &group.params;
        if params.is_empty() {
            return Ok(());
        }
        // NOTE: this assumes all params in the group have the same shape
        let shape = params[0].dims().to_vec();
        let (m, n) = (shape[shape.len() - 2], shape[shape.len() - 1]);
        let num_params = params.len();
        let device = params[0].device().clone();
        let dtype = params[0].dtype();

        let state = match muon_state.entry(group_idx) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                // Second momentum buffer is factored, either per-row or per-column
                let state_shape = if m >= n {
                    (num_params, m, 1)
                } else {
                    (num_params, 1, n)
                };
                e.insert(MuonState {
                    momentum_buffer: Tensor::zeros((num_params, m, n), dtype, &device)?,
                    second_momentum_buffer: Tensor::zeros(state_shape, dtype, &device)?,
                })
            }
        };
        let red_dim = if m >= n { 2 } else { 1 };

        // Stack grads and params
        let grads: Vec<Tensor> = params
            .iter()
            .map(|p| {
                get_grad(p).ok_or_else(|| {
                    candle_core::Error::Msg("missing gradient for a Muon parameter".into())
                })
            })
            .collect::<Result<_>>()?;
        let stacked_grads = Tensor::stack(&grads, 0)?;
        let stacked_params = Tensor::stack(
            &params
                .iter()
                .map(|p| p.as_tensor().clone())
                .collect::<Vec<_>>(),
            0,
        )?;

        // Boost the LR for tall matrices
        let lr = group.lr * (m as f64 / n as f64).max(1.0).sqrt();

        let (new_params, momentum_buffer, second_momentum_buffer) = muon_step(
            &stacked_grads,
            &stacked_params,
            &state.momentum_buffer,
            &state.second_momentum_buffer,
            group.momentum,
            lr,
            group.weight_decay,
            group.beta2,
            group.ns_steps,
            red_dim,
        )?;
        state.momentum_buffer = momentum_buffer;
        state.second_momentum_buffer = second_momentum_buffer;

        // Copy back to original params
        for (i, p) in params.iter().enumerate() {
            p.set(&new_params.i(i)?)?;
        }
        Ok(())
    }

    /// Apply one optimization step; get_grad maps each parameter to its
    /// (possibly accumulated) gradient.
    pub fn step_with(&mut self, get_grad: &dyn Fn(&Var) -> Option<Tensor>) -> Result<()> {
        for (group_idx, group) in self.param_groups.iter().enumerate() {
            match group.kind {
                GroupKind::AdamW => {
                    Self::step_adamw(group_idx, group, &mut self.adamw_state, get_grad)?
                }
                GroupKind::Muon => {
                    Self::step_muon(group_idx, group, &mut self.muon_state, get_grad)?
                }
            }
        }
        Ok(())
    }

    /// Convenience for the common case of a single backward pass.
    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.step_with(&|p: &Var| grads.get(p).cloned())
    }

    /// Optimizer state as named tensors (for checkpointing). Step counts are
    /// stored as 1-element i64 tensors.
    pub fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        let device = candle_core::Device::Cpu;
        for ((gi, pi), st) in &self.adamw_state {
            out.insert(
                format!("adamw.{gi}.{pi}.step"),
                Tensor::from_vec(vec![st.step as i64], 1, &device)?,
            );
            out.insert(format!("adamw.{gi}.{pi}.exp_avg"), st.exp_avg.clone());
            out.insert(format!("adamw.{gi}.{pi}.exp_avg_sq"), st.exp_avg_sq.clone());
        }
        for (gi, st) in &self.muon_state {
            out.insert(
                format!("muon.{gi}.momentum_buffer"),
                st.momentum_buffer.clone(),
            );
            out.insert(
                format!("muon.{gi}.second_momentum_buffer"),
                st.second_momentum_buffer.clone(),
            );
        }
        Ok(out)
    }

    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> Result<()> {
        self.adamw_state.clear();
        self.muon_state.clear();
        for (key, tensor) in state {
            let parts: Vec<&str> = key.split('.').collect();
            match parts.as_slice() {
                ["adamw", gi, pi, field] => {
                    let gi: usize = gi
                        .parse()
                        .map_err(|_| candle_core::Error::Msg(format!("bad key {key}")))?;
                    let pi: usize = pi
                        .parse()
                        .map_err(|_| candle_core::Error::Msg(format!("bad key {key}")))?;
                    let group = &self.param_groups[gi];
                    let entry = self.adamw_state.entry((gi, pi)).or_insert(AdamWState {
                        step: 0,
                        exp_avg: group.params[pi].zeros_like()?,
                        exp_avg_sq: group.params[pi].zeros_like()?,
                    });
                    match *field {
                        "step" => entry.step = tensor.to_vec1::<i64>()?[0] as u32,
                        "exp_avg" => entry.exp_avg = tensor.clone(),
                        "exp_avg_sq" => entry.exp_avg_sq = tensor.clone(),
                        _ => candle_core::bail!("unknown state key: {key}"),
                    }
                }
                ["muon", gi, field] => {
                    let gi: usize = gi
                        .parse()
                        .map_err(|_| candle_core::Error::Msg(format!("bad key {key}")))?;
                    let zeros = tensor.zeros_like()?;
                    let entry = self.muon_state.entry(gi).or_insert(MuonState {
                        momentum_buffer: zeros.clone(),
                        second_momentum_buffer: zeros,
                    });
                    match *field {
                        "momentum_buffer" => entry.momentum_buffer = tensor.clone(),
                        "second_momentum_buffer" => entry.second_momentum_buffer = tensor.clone(),
                        _ => candle_core::bail!("unknown state key: {key}"),
                    }
                }
                _ => candle_core::bail!("unknown state key: {key}"),
            }
        }
        Ok(())
    }
}
