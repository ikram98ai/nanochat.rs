//! Tests for the BPE tokenizer against the tiny trained tokenizer and the
//! expected encodings in tests/data (tokenizer.json, tokenizer_fixtures.json).
//!
//! tests/data is generated, not checked in: `cargo run --release --example
//! gen_test_data` rebuilds it from this implementation, so these tests detect
//! behaviour changes since it was generated. The Python reference's fixture
//! scripts write the same layout, which turns them into parity tests.

use std::path::PathBuf;

use nanochat::tokenizer::{Tokenizer, SPECIAL_TOKENS};

fn data_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data");
    assert!(dir.join("meta.json").exists(), "tests/data is generated, not checked in: run `cargo run --release --example gen_test_data` first");
    dir
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

/// Training on the same corpus must reproduce the Python-trained tokenizer:
/// identical merges (incl. tie-breaking) and identical encodings.
#[test]
fn test_train_from_iterator_parity() {
    let corpus = [
        "Hello world! This is a tiny training corpus for the nanochat tokenizer. ",
        "The quick brown fox jumps over the lazy dog. 0123456789 42 100 2048. ",
        "I'd've thought it'll be fine, isn't it? We're sure they've gone. ",
        "def main():\n    print('hello')\n    return 0\n",
        "Numbers: 1, 22, 333, 4444. Punctuation?! (parens) [brackets] {braces}. ",
    ];
    let texts = corpus.iter().cycle().take(corpus.len() * 20).map(|s| s.to_string());
    let trained = Tokenizer::train_from_iterator(texts, 300).unwrap();
    let reference = load_tokenizer();

    assert_eq!(trained.get_vocab_size(), reference.get_vocab_size(), "vocab size");
    // every token id decodes to the same bytes => identical merge tables
    for id in 0..trained.get_vocab_size() as u32 {
        assert_eq!(trained.id_to_token(id), reference.id_to_token(id), "token {id}");
    }
    let fx = load_fixtures();
    for case in fx["encode_cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        assert_eq!(
            trained.encode(text, None, None),
            as_u32s(&case["ids"]),
            "trained encode({text:?})"
        );
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

/// The reference's tokenizer.pkl and a HuggingFace-format tokenizer.json of the
/// same tokenizer load directly and behave identically to the native format.
#[test]
fn test_foreign_tokenizer_formats() {
    let native = load_tokenizer();
    let fx = load_fixtures();
    for format in ["pkl", "hf"] {
        let dir = data_dir().join("tokenizer_formats").join(format);
        let tokenizer = Tokenizer::from_directory(&dir).unwrap();
        assert_eq!(tokenizer.get_vocab_size(), native.get_vocab_size(), "{format}: vocab size");
        assert_eq!(tokenizer.get_bos_token_id(), native.get_bos_token_id(), "{format}: bos");
        assert_eq!(tokenizer.token_bytes(), native.token_bytes(), "{format}: token bytes");
        let mut specials = tokenizer.get_special_tokens();
        specials.sort();
        let mut native_specials = native.get_special_tokens();
        native_specials.sort();
        assert_eq!(specials, native_specials, "{format}: special tokens");
        for name in &specials {
            assert_eq!(tokenizer.encode_special(name), native.encode_special(name), "{format}: {name}");
        }
        for id in 0..native.get_vocab_size() as u32 {
            assert_eq!(
                tokenizer.decode_single_token_bytes(id),
                native.decode_single_token_bytes(id),
                "{format}: bytes of token {id}"
            );
        }
        for case in fx["encode_cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            assert_eq!(tokenizer.encode(text, None, None), as_u32s(&case["ids"]), "{format}: encode {text:?}");
        }
    }
}
