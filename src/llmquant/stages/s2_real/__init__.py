"""Phase 2 -- real quant reference (not implemented yet).

Will hold RealQuantLinear: int weights kept as integers, with the matmul done by verified
PyTorch ops. Pass criterion: its PPL matches the fake-quant PPL to float precision; a
larger gap means a bug.

With group_size=128 a group's partial sum peaks near 2.06M, inside fp32's exact-integer
range, so both W4A8 and W8A8 can use a plain fp32 matmul and stay bit-exact (TF32 off).
"""
