//! Read-only GGUF reader.
//!
//! The file is memory-mapped and every tensor is handed back as a slice of that
//! map, so opening a 2.7 GB model costs no RAM and no copy. Metadata arrays are
//! recorded as offsets instead of being materialized during parsing: the 130k
//! token strings and 129k merges are only decoded when the tokenizer asks.

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;

use crate::error::EngineError;

const MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

/// Tensor element type, limited to what this model actually uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Q8_0,
}

impl DType {
    fn from_id(id: u32) -> Result<Self, EngineError> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            8 => Ok(Self::Q8_0),
            other => Err(EngineError::Unsupported(format!("tensor type id {other}"))),
        }
    }

    /// Elements per quantization block.
    pub fn block_elements(self) -> u64 {
        match self {
            Self::Q8_0 => 32,
            Self::F32 | Self::F16 => 1,
        }
    }

    /// Bytes per quantization block.
    pub fn block_bytes(self) -> u64 {
        match self {
            Self::Q8_0 => 34,
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    pub fn byte_len(self, elements: u64) -> u64 {
        elements / self.block_elements() * self.block_bytes()
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: DType,
    pub offset: u64,
    pub byte_len: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

#[derive(Debug, Clone)]
enum MetaValue {
    UInt(u64),
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    /// Array payload as an absolute file offset plus element count and type id.
    Array {
        elem_type: u32,
        len: u64,
        data_offset: u64,
    },
}

pub struct GgufFile {
    map: Mmap,
    meta: HashMap<String, MetaValue>,
    tensors: Vec<TensorInfo>,
    data_start: u64,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, EngineError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the map is only ever read; the model file is never mutated while open.
        let map = unsafe { Mmap::map(&file)? };
        let mut reader = Reader::new(&map);
        let mut meta = HashMap::new();
        let mut tensors = Vec::new();

        let magic = reader.u32()?;
        if magic != MAGIC {
            return Err(EngineError::Format(format!("bad magic {magic:#010x}")));
        }
        let version = reader.u32()?;
        if !(2..=3).contains(&version) {
            return Err(EngineError::Format(format!(
                "unsupported version {version}"
            )));
        }
        let tensor_count = reader.u64()?;
        let kv_count = reader.u64()?;

        for _ in 0..kv_count {
            let key = reader.str()?.to_string();
            let value = read_meta_value(&mut reader)?;
            meta.insert(key, value);
        }

        for _ in 0..tensor_count {
            let name = reader.str()?.to_string();
            let n_dims = reader.u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(reader.u64()?);
            }
            let dtype = DType::from_id(reader.u32()?)?;
            let offset = reader.u64()?;
            let elements: u64 = dims.iter().product();
            if !elements.is_multiple_of(dtype.block_elements()) {
                return Err(EngineError::Format(format!(
                    "tensor {name}: {elements} elements is not a multiple of the {} block size",
                    dtype.block_elements()
                )));
            }
            tensors.push(TensorInfo {
                name,
                dims,
                dtype,
                offset,
                byte_len: dtype.byte_len(elements),
            });
        }

        let alignment = match meta.get("general.alignment") {
            Some(MetaValue::UInt(value)) => *value,
            _ => DEFAULT_ALIGNMENT,
        };
        let data_start = reader.pos().div_ceil(alignment) * alignment;
        if data_start > map.len() as u64 {
            return Err(EngineError::Format(
                "tensor data starts past end of file".into(),
            ));
        }
        for tensor in &tensors {
            if data_start + tensor.offset + tensor.byte_len > map.len() as u64 {
                return Err(EngineError::Format(format!(
                    "tensor {} runs past end of file",
                    tensor.name
                )));
            }
        }

        Ok(Self {
            map,
            meta,
            tensors,
            data_start,
        })
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, EngineError> {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| EngineError::MissingTensor(name.to_string()))
    }

    /// Raw little-endian bytes of a tensor, borrowed from the mmap.
    pub fn data(&self, name: &str) -> Result<&[u8], EngineError> {
        let tensor = self.tensor(name)?;
        let start = (self.data_start + tensor.offset) as usize;
        Ok(&self.map[start..start + tensor.byte_len as usize])
    }

    pub fn meta_u64(&self, key: &str) -> Result<u64, EngineError> {
        match self.meta.get(key) {
            Some(MetaValue::UInt(value)) => Ok(*value),
            Some(MetaValue::Int(value)) if *value >= 0 => Ok(*value as u64),
            Some(_) => Err(EngineError::Format(format!(
                "{key} is not an unsigned integer"
            ))),
            None => Err(EngineError::MissingMetadata(key.to_string())),
        }
    }

    pub fn meta_f32(&self, key: &str) -> Result<f32, EngineError> {
        match self.meta.get(key) {
            Some(MetaValue::Float(value)) => Ok(*value as f32),
            Some(_) => Err(EngineError::Format(format!("{key} is not a float"))),
            None => Err(EngineError::MissingMetadata(key.to_string())),
        }
    }

    pub fn meta_bool(&self, key: &str) -> Result<bool, EngineError> {
        match self.meta.get(key) {
            Some(MetaValue::Bool(value)) => Ok(*value),
            Some(_) => Err(EngineError::Format(format!("{key} is not a bool"))),
            None => Err(EngineError::MissingMetadata(key.to_string())),
        }
    }

    /// Same as [`Self::meta_u64`], but a missing key is not an error. Used for
    /// fields a GGUF may leave out and every loader fills with a default.
    pub fn meta_u64_opt(&self, key: &str) -> Option<u64> {
        match self.meta.get(key) {
            Some(MetaValue::UInt(value)) => Some(*value),
            Some(MetaValue::Int(value)) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    pub fn meta_f32_opt(&self, key: &str) -> Option<f32> {
        match self.meta.get(key) {
            Some(MetaValue::Float(value)) => Some(*value as f32),
            _ => None,
        }
    }

    /// Whether any key starts with `prefix`, which is how a model declares a
    /// block type this engine does not implement.
    pub fn has_key_prefix(&self, prefix: &str) -> bool {
        self.meta.keys().any(|key| key.starts_with(prefix))
    }

    pub fn meta_str(&self, key: &str) -> Result<&str, EngineError> {
        match self.meta.get(key) {
            Some(MetaValue::Text(value)) => Ok(value),
            Some(_) => Err(EngineError::Format(format!("{key} is not a string"))),
            None => Err(EngineError::MissingMetadata(key.to_string())),
        }
    }

    /// Decode an integer array (token types) straight out of the map.
    pub fn meta_i32_array(&self, key: &str) -> Result<Vec<i32>, EngineError> {
        let (len, data_offset) = match self.meta.get(key) {
            Some(MetaValue::Array {
                elem_type: 4,
                len,
                data_offset,
            }) => (*len, *data_offset),
            Some(MetaValue::Array {
                elem_type: 5,
                len,
                data_offset,
            }) => (*len, *data_offset),
            Some(MetaValue::Array { .. }) => {
                return Err(EngineError::Format(format!(
                    "{key} is not an integer array"
                )));
            }
            Some(_) => return Err(EngineError::Format(format!("{key} is not an array"))),
            None => return Err(EngineError::MissingMetadata(key.to_string())),
        };
        let mut reader = Reader::at(&self.map, data_offset as usize);
        let mut values = Vec::with_capacity(len as usize);
        for _ in 0..len {
            values.push(reader.u32()? as i32);
        }
        Ok(values)
    }

    /// Decode a string array (tokens, merges) straight out of the map.
    pub fn meta_str_array(&self, key: &str) -> Result<Vec<&str>, EngineError> {
        let (len, data_offset) = match self.meta.get(key) {
            Some(MetaValue::Array {
                elem_type: 8,
                len,
                data_offset,
            }) => (*len, *data_offset),
            Some(MetaValue::Array { .. }) => {
                return Err(EngineError::Format(format!("{key} is not a string array")));
            }
            Some(_) => return Err(EngineError::Format(format!("{key} is not an array"))),
            None => return Err(EngineError::MissingMetadata(key.to_string())),
        };
        let mut reader = Reader::at(&self.map, data_offset as usize);
        let mut values = Vec::with_capacity(len as usize);
        for _ in 0..len {
            values.push(reader.str()?);
        }
        Ok(values)
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn at(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos }
    }

    fn pos(&self) -> u64 {
        self.pos as u64
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|end| *end <= self.data.len())
            .ok_or_else(|| EngineError::Format("unexpected end of file".into()))?;
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn skip(&mut self, len: u64) -> Result<(), EngineError> {
        let len = usize::try_from(len)
            .map_err(|_| EngineError::Format("array too large for this platform".into()))?;
        self.take(len).map(|_| ())
    }

    fn u8(&mut self) -> Result<u8, EngineError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, EngineError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, EngineError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, EngineError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, EngineError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, EngineError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64, EngineError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn str(&mut self) -> Result<&'a str, EngineError> {
        let len = self.u64()?;
        let bytes = self.skip_exact(len)?;
        std::str::from_utf8(bytes)
            .map_err(|_| EngineError::Format("invalid utf-8 in string".into()))
    }

    fn skip_exact(&mut self, len: u64) -> Result<&'a [u8], EngineError> {
        let len = usize::try_from(len)
            .map_err(|_| EngineError::Format("string too large for this platform".into()))?;
        self.take(len)
    }
}

fn read_meta_value(reader: &mut Reader<'_>) -> Result<MetaValue, EngineError> {
    let type_id = reader.u32()?;
    read_typed_value(reader, type_id)
}

fn read_typed_value(reader: &mut Reader<'_>, type_id: u32) -> Result<MetaValue, EngineError> {
    Ok(match type_id {
        0 => MetaValue::UInt(reader.u8()? as u64),
        1 => MetaValue::Int(reader.u8()? as i8 as i64),
        2 => MetaValue::UInt(reader.u16()? as u64),
        3 => MetaValue::Int(reader.u16()? as i16 as i64),
        4 => MetaValue::UInt(reader.u32()? as u64),
        5 => MetaValue::Int(reader.u32()? as i32 as i64),
        6 => MetaValue::Float(reader.f32()? as f64),
        7 => MetaValue::Bool(reader.u8()? != 0),
        8 => MetaValue::Text(reader.str()?.to_string()),
        9 => {
            let elem_type = reader.u32()?;
            let len = reader.u64()?;
            let data_offset = reader.pos();
            skip_array(reader, elem_type, len)?;
            MetaValue::Array {
                elem_type,
                len,
                data_offset,
            }
        }
        10 => MetaValue::UInt(reader.u64()?),
        11 => MetaValue::Int(reader.i64()?),
        12 => MetaValue::Float(reader.f64()?),
        other => {
            return Err(EngineError::Format(format!(
                "unknown metadata value type {other}"
            )));
        }
    })
}

fn skip_array(reader: &mut Reader<'_>, elem_type: u32, len: u64) -> Result<(), EngineError> {
    let fixed_size = match elem_type {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        8 | 9 => None,
        other => {
            return Err(EngineError::Format(format!(
                "unknown array element type {other}"
            )));
        }
    };
    match fixed_size {
        Some(size) => reader.skip(size * len),
        None => {
            for _ in 0..len {
                if elem_type == 8 {
                    reader.str()?;
                } else {
                    let nested_type = reader.u32()?;
                    let nested_len = reader.u64()?;
                    skip_array(reader, nested_type, nested_len)?;
                }
            }
            Ok(())
        }
    }
}
