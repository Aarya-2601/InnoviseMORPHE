"""MORPHE storage module."""
from morphe.storage.ledger import VersionLedger
from morphe.storage.record_store import Record, RecordStore

__all__ = ["Record", "RecordStore", "VersionLedger"]

