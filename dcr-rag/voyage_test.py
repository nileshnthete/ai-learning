import numpy as np
import voyageai

vo = voyageai.Client()  # reads VOYAGE_API_KEY

docs = [
    "Snowflake supports key-pair authentication for service users.",
    "Strawberries are on sale at No Frills for $2.99.",
    "Databricks service principals can use OAuth machine-to-machine tokens.",
]
doc_vecs = vo.embed(docs, model="voyage-4", input_type="document").embeddings
q_vec = vo.embed(["How do I authenticate a service account?"], model="voyage-4",
                 input_type="query").embeddings[0]

scores = np.dot(doc_vecs, q_vec)  # vectors are normalized, so dot product = similarity
for score, doc in sorted(zip(scores, docs), reverse=True):
    print(f"{score:.3f}  {doc}")