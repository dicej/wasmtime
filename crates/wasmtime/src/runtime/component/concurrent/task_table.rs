// TODO: This duplicates a lot of resource_table.rs; consider reducing that duplication.

use super::{Task, TaskId};
use std::collections::BTreeSet;

#[derive(Debug)]
/// Errors returned by operations on `TaskTable`
pub enum TaskTableError {
    /// TaskTable has no free keys
    Full,
    /// Task not present in table
    NotPresent,
    /// Task cannot be deleted because child tasks exist in the table.
    HasChildren,
}

impl std::fmt::Display for TaskTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "task table has no free keys"),
            Self::NotPresent => write!(f, "task not present"),
            Self::HasChildren => write!(f, "task has children"),
        }
    }
}
impl std::error::Error for TaskTableError {}

/// The `TaskTable` type maps a `TaskId` to its `Task`.
pub struct TaskTable<T> {
    entries: Vec<Entry<T>>,
    free_head: Option<usize>,
}

enum Entry<T> {
    Free { next: Option<usize> },
    Occupied { entry: TableEntry<T> },
}

impl<T> Entry<T> {
    pub fn occupied(&self) -> Option<&TableEntry<T>> {
        match self {
            Self::Occupied { entry } => Some(entry),
            Self::Free { .. } => None,
        }
    }

    pub fn occupied_mut(&mut self) -> Option<&mut TableEntry<T>> {
        match self {
            Self::Occupied { entry } => Some(entry),
            Self::Free { .. } => None,
        }
    }
}

/// This structure tracks parent and child relationships for a given table entry.
///
/// Parents and children are referred to by table index. We maintain the
/// following invariants to prevent orphans and cycles:
/// * parent can only be assigned on creating the entry.
/// * parent, if some, must exist when creating the entry.
/// * whenever a child is created, its index is added to children.
/// * whenever a child is deleted, its index is removed from children.
/// * an entry with children may not be deleted.
struct TableEntry<T> {
    /// The entry in the table
    task: Task<T>,
    /// The index of the parent of this entry, if it has one.
    parent: Option<u32>,
    /// The indicies of any children of this entry.
    children: BTreeSet<u32>,
}

impl<T> TableEntry<T> {
    fn new(task: Task<T>, parent: Option<u32>) -> Self {
        Self {
            task,
            parent,
            children: BTreeSet::new(),
        }
    }
    fn add_child(&mut self, child: u32) {
        debug_assert!(!self.children.contains(&child));
        self.children.insert(child);
    }
    fn remove_child(&mut self, child: u32) {
        let was_removed = self.children.remove(&child);
        debug_assert!(was_removed);
    }
}

impl<T> TaskTable<T> {
    /// Create an empty table
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            free_head: None,
        }
    }

    /// Inserts a new task into this table, returning a corresponding
    /// `TaskId<T>` which can be used to refer to it after it was inserted.
    pub fn push(&mut self, task: Task<T>) -> Result<TaskId<T>, TaskTableError> {
        let idx = self.push_(TableEntry::new(task, None))?;
        Ok(TaskId::new(idx))
    }

    /// Pop an index off of the free list, if it's not empty.
    fn pop_free_list(&mut self) -> Option<usize> {
        if let Some(ix) = self.free_head {
            // Advance free_head to the next entry if one is available.
            match &self.entries[ix] {
                Entry::Free { next } => self.free_head = *next,
                Entry::Occupied { .. } => unreachable!(),
            }
            Some(ix)
        } else {
            None
        }
    }

    /// Free an entry in the table, returning its [`TableEntry`]. Add the index to the free list.
    fn free_entry(&mut self, ix: usize) -> TableEntry<T> {
        let entry = match std::mem::replace(
            &mut self.entries[ix],
            Entry::Free {
                next: self.free_head,
            },
        ) {
            Entry::Occupied { entry } => entry,
            Entry::Free { .. } => unreachable!(),
        };

        self.free_head = Some(ix);

        entry
    }

    /// Push a new entry into the table, returning its handle. This will prefer to use free entries
    /// if they exist, falling back on pushing new entries onto the end of the table.
    fn push_(&mut self, e: TableEntry<T>) -> Result<u32, TaskTableError> {
        if let Some(free) = self.pop_free_list() {
            self.entries[free] = Entry::Occupied { entry: e };
            Ok(free as u32)
        } else {
            let ix = self
                .entries
                .len()
                .try_into()
                .map_err(|_| TaskTableError::Full)?;
            self.entries.push(Entry::Occupied { entry: e });
            Ok(ix)
        }
    }

    fn occupied(&self, key: u32) -> Result<&TableEntry<T>, TaskTableError> {
        self.entries
            .get(key as usize)
            .and_then(Entry::occupied)
            .ok_or(TaskTableError::NotPresent)
    }

    fn occupied_mut(&mut self, key: u32) -> Result<&mut TableEntry<T>, TaskTableError> {
        self.entries
            .get_mut(key as usize)
            .and_then(Entry::occupied_mut)
            .ok_or(TaskTableError::NotPresent)
    }

    /// Insert a task at the next available index, and track that it has a
    /// parent task.
    ///
    /// The parent must exist to create a child. All child tasks must be
    /// destroyed before a parent can be destroyed - otherwise
    /// [`TaskTable::delete`] will fail with [`TaskTableError::HasChildren`].
    ///
    /// Parent-child relationships are tracked inside the table to ensure that a
    /// parent is not deleted while it has live children. This allows children
    /// to hold "references" to a parent by table index, to avoid needing
    /// e.g. an `Arc<Mutex<parent>>` and the associated locking overhead and
    /// design issues, such as child existence extending lifetime of parent
    /// referent even after parent is destroyed, possibility for deadlocks.
    ///
    /// Parent-child relationships may not be modified once created. There
    /// is no way to observe these relationships through the [`TaskTable`]
    /// methods except for erroring on deletion, or the [`std::fmt::Debug`]
    /// impl.
    pub fn push_child(
        &mut self,
        task: Task<T>,
        parent: TaskId<T>,
    ) -> Result<TaskId<T>, TaskTableError> {
        let parent = parent.rep();
        self.occupied(parent)?;
        let child = self.push_(TableEntry::new(task, Some(parent)))?;
        self.occupied_mut(parent)?.add_child(child);
        Ok(TaskId::new(child))
    }

    /// Get an immutable reference to a task of a given type at a given index.
    ///
    /// Multiple shared references can be borrowed at any given time.
    pub fn get(&self, key: TaskId<T>) -> Result<&Task<T>, TaskTableError> {
        self.get_(key.rep())
    }

    fn get_(&self, key: u32) -> Result<&Task<T>, TaskTableError> {
        let r = self.occupied(key)?;
        Ok(&r.task)
    }

    /// Get an mutable reference to a task of a given type at a given index.
    pub fn get_mut(&mut self, key: TaskId<T>) -> Result<&mut Task<T>, TaskTableError> {
        self.get_mut_(key.rep())
    }

    pub fn get_mut_(&mut self, key: u32) -> Result<&mut Task<T>, TaskTableError> {
        let r = self.occupied_mut(key)?;
        Ok(&mut r.task)
    }

    /// Delete the specified task
    pub fn delete(&mut self, key: TaskId<T>) -> Result<Task<T>, TaskTableError> {
        Ok(self.delete_entry(key.rep())?.task)
    }

    fn delete_entry(&mut self, key: u32) -> Result<TableEntry<T>, TaskTableError> {
        if !self.occupied(key)?.children.is_empty() {
            return Err(TaskTableError::HasChildren);
        }
        let e = self.free_entry(key as usize);
        if let Some(parent) = e.parent {
            // Remove deleted task from parent's child list.  Parent must still
            // be present because it cant be deleted while still having
            // children:
            self.occupied_mut(parent)
                .expect("missing parent")
                .remove_child(key);
        }
        Ok(e)
    }
}

impl<T> Default for TaskTable<T> {
    fn default() -> Self {
        Self::new()
    }
}
