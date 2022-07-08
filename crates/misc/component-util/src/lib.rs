#[derive(Debug, Copy, Clone)]
pub enum DiscriminantSize {
    Size1,
    Size2,
    Size4,
}

impl DiscriminantSize {
    pub fn from_count(count: usize) -> Option<Self> {
        if count <= 0xFF {
            Some(Self::Size1)
        } else if count <= 0xFFFF {
            Some(Self::Size2)
        } else if count <= 0xFFFF_FFFF {
            Some(Self::Size4)
        } else {
            None
        }
    }
}

impl From<DiscriminantSize> for u32 {
    fn from(size: DiscriminantSize) -> u32 {
        match size {
            DiscriminantSize::Size1 => 1,
            DiscriminantSize::Size2 => 2,
            DiscriminantSize::Size4 => 4,
        }
    }
}

impl From<DiscriminantSize> for usize {
    fn from(size: DiscriminantSize) -> usize {
        match size {
            DiscriminantSize::Size1 => 1,
            DiscriminantSize::Size2 => 2,
            DiscriminantSize::Size4 => 4,
        }
    }
}

pub enum FlagsSize {
    /// Flags can fit in a u8
    Size1,
    /// Flags can fit in a u16
    Size2,
    /// Flags can fit in a specified number of u32 fields
    Size4Plus(usize),
}

impl FlagsSize {
    pub fn from_count(count: usize) -> FlagsSize {
        if count <= 8 {
            FlagsSize::Size1
        } else if count <= 16 {
            FlagsSize::Size2
        } else {
            FlagsSize::Size4Plus(ceiling_divide(count, 32))
        }
    }
}

pub fn ceiling_divide(n: usize, d: usize) -> usize {
    (n + d - 1) / d
}
