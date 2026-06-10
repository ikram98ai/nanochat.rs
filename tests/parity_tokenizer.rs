//! Parity tests for the BPE tokenizer against the tiny tokenizer trained and
//! exported by the Python implementation (tests/gen_reference.py).

use std::path::PathBuf;

use nanochat::tokenizer::{Tokenizer, SPECIAL_TOKENS};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data")
}

fn load_tokenizer() -> Tokenizer {
    Tokenizer::from_directory(&data_dir()).unwrap()
}

fn load_fixtures() -> serde_json::Value {
    let data = std::fs::read_to_string(data_dir().join("tokenizer_fixtures.json")).unwrap();
    serde_json::from_str(&data).unwrap()
}

fn as_u32s(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn as_u8s(v: &serde_json::Value) -> Vec<u8> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect()
}

#[test]
fn test_vocab_and_special_tokens() {
    let tok = load_tokenizer();
    let fx = load_fixtures();
    assert_eq!(tok.get_vocab_size() as u64, fx["vocab_size"].as_u64().unwrap());
    assert_eq!(tok.get_bos_token_id() as u64, fx["bos_token_id"].as_u64().unwrap());
    for name in SPECIAL_TOKENS {
        assert_eq!(
            tok.encode_special(name) as u64,
            fx["special_tokens"][name].as_u64().unwrap(),
            "special token {name}"
        );
    }
}

#[test]
fn test_encode_decode() {
    let tok = load_tokenizer();
    let fx = load_fixtures();
    for case in fx["encode_cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected_ids = as_u32s(&case["ids"]);
        let ids = tok.encode(text, None, None);
        assert_eq!(ids, expected_ids, "encode({text:?})");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, case["decoded"].as_str().unwrap(), "decode round-trip of {text:?}");
    }
}

#[test]
fn test_encode_prepend_append() {
    let tok = load_tokenizer();
    let fx = load_fixtures();
    let case = &fx["encode_prepend_append"];
    let ids = tok.encode(case["text"].as_str().unwrap(), Some("<|bos|>"), Some("<|assistant_end|>"));
    assert_eq!(ids, as_u32s(&case["ids"]));
}

#[test]
fn test_encode_batch() {
    let tok = load_tokenizer();
    let fx = load_fixtures();
    let expected: Vec<Vec<u32>> = fx["encode_batch"].as_array().unwrap().iter().map(as_u32s).collect();
    let ids = tok.encode_batch(&["Hello world", "foo bar"]);
    assert_eq!(ids, expected);
}

#[test]
fn test_render_conversation() {
    let tok = load_tokenizer();
    let fx = load_fixtures();
    let case = &fx["render_conversation"];
    let (ids, mask) = tok.render_conversation(&case["conversation"], 2048).unwrap();
    assert_eq!(ids, as_u32s(&case["ids"]), "render_conversation ids");
    assert_eq!(mask, as_u8s(&case["mask"]), "render_conversation mask");

    // truncation
    let case_trunc = &fx["render_conversation_truncated"];
    let max_tokens = case_trunc["max_tokens"].as_u64().unwrap() as usize;
    let (ids, mask) = tok.render_conversation(&case["conversation"], max_tokens).unwrap();
    assert_eq!(ids, as_u32s(&case_trunc["ids"]), "truncated ids");
    assert_eq!(mask, as_u8s(&case_trunc["mask"]), "truncated mask");
}

#[test]
fn test_render_for_completion() {
    let tok = load_tokenizer();
    let fx = load_fixtures();
    let case = &fx["render_for_completion"];
    let ids = tok.render_for_completion(&case["conversation"]).unwrap();
    assert_eq!(ids, as_u32s(&case["ids"]));
}

#[test]
fn test_id_to_token() {
    let tok = load_tokenizer();
    let fx = load_fixtures();
    for (id, expected) in fx["id_to_token_samples"].as_object().unwrap() {
        let id: u32 = id.parse().unwrap();
        assert_eq!(tok.id_to_token(id), expected.as_str().unwrap(), "id_to_token({id})");
    }
}

#[test]
fn test_save_load_roundtrip() {
    let tok = load_tokenizer();
    let dir = std::env::temp_dir().join("nanochat_tokenizer_roundtrip");
    tok.save(&dir).unwrap();
    let tok2 = Tokenizer::from_directory(&dir).unwrap();
    let text = "Hello world, this is a round-trip test! 1234";
    assert_eq!(tok.encode(text, None, None), tok2.encode(text, None, None));
    assert_eq!(tok.get_vocab_size(), tok2.get_vocab_size());
}
