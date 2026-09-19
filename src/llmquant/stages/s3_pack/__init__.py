"""Phase 3 -- packing and the .bin format (not implemented yet).

Will hold the nibble packing, the single-file [magic][header][tensors] writer and the
np.memmap loader. Pass criterion: the integers and scales read back are bit-exact with
Phase 2. See "packing 저장 형식" in docs/w4a8_rtn_notes.md for the confirmed layout.
"""
