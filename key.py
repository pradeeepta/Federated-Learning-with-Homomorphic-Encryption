# generate_ckks_full_context.py
import tenseal as ts

def create_ckks_context():
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2 ** 40
    context.generate_galois_keys()
    return context

ctx = create_ckks_context()

# Save full context (including secret key)
with open("ckks_context_full.tenseal", "wb") as f:
    f.write(ctx.serialize(save_secret_key=True))

# Also save public-only context for clients
with open("ckks_context.tenseal", "wb") as f:
    f.write(ctx.serialize(save_secret_key=False))

print("✅ CKKS context files created.")
