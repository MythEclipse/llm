//! GPT-2 byte-level alphabet: a bijection between the 256 bytes and printable
//! unicode code points, so any byte sequence is a legal BPE input.

use std::collections::HashMap;

pub struct ByteAlphabet {
    to_char: [char; 256],
    to_byte: HashMap<char, u8>,
}

impl ByteAlphabet {
    pub fn new() -> Self {
        // Printable bytes keep their own code point; the rest are pushed into the
        // private-use range 256.. in ascending byte order, exactly like GPT-2.
        let mut printable = Vec::with_capacity(256);
        printable.extend(33u16..=126);
        printable.extend(161u16..=172);
        printable.extend(174u16..=255);

        let mut to_char = ['\0'; 256];
        let mut to_byte = HashMap::with_capacity(256);
        let mut spare = 256u32;
        for (byte, slot) in to_char.iter_mut().enumerate() {
            let code_point = if printable.contains(&(byte as u16)) {
                byte as u32
            } else {
                let assigned = spare;
                spare += 1;
                assigned
            };
            let character = char::from_u32(code_point).expect("byte alphabet stays in range");
            *slot = character;
            to_byte.insert(character, byte as u8);
        }
        Self { to_char, to_byte }
    }

    pub fn char_for(&self, byte: u8) -> char {
        self.to_char[byte as usize]
    }

    pub fn byte_for(&self, character: char) -> Option<u8> {
        self.to_byte.get(&character).copied()
    }
}

impl Default for ByteAlphabet {
    fn default() -> Self {
        Self::new()
    }
}
