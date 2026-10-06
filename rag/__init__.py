"""The RAG pipeline, one file per step.

step1_load        -> read the tender PDF and tidy the text
step2_chunk       -> cut the text into overlapping chunks
step3_embed_store -> turn chunks into vectors and save them
step4_retrieve    -> find the chunks closest to the question
step5_prompt      -> put those chunks into the prompt
step6_generate    -> send the prompt to the LLM and get the answer
"""
