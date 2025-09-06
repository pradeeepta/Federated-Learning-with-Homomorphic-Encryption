# generate_ckks_context.py
import tenseal as ts

def create_ckks_context():
    context = ts.context(
        scheme=ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

context = create_ckks_context()

# ✅ Save context WITHOUT secret key for clients
with open("ckks_context.tenseal", "wb") as f:
    f.write(context.serialize(save_secret_key=False))
print("✅ Public CKKS context saved as 'ckks_context.tenseal' (for clients)")

# ✅ Save context WITH secret key for server
with open("ckks_context_full.tenseal", "wb") as f:
    f.write(context.serialize(save_secret_key=True))
print("✅ Private CKKS context saved as 'ckks_context_full.tenseal' (for server)")
