//! Dataset loader scaffolding placeholder for the workspace split.

/// Minimal dataset descriptor with just a name for now.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DatasetDescriptor {
    /// Human readable dataset identifier.
    pub name: String,
}

impl DatasetDescriptor {
    /// Creates a descriptor with the provided name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::DatasetDescriptor;

    #[test]
    fn descriptor_holds_name() {
        let descriptor = DatasetDescriptor::new("xor");
        assert_eq!(descriptor.name, "xor");
    }
}
