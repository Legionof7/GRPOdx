doc_outs = doctor_model.generate(
    [doc_input],
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
)
