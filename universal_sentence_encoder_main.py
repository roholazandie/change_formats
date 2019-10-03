import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import numpy as np
import time


class UniversalSentenceEncoder():

    def __init__(self):
        #module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-lite/2"
        #module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        spm_path = self.session.run(self.embed(signature="spm_path"))
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)

    def ecnode_keras(self):
        '''
        only works with tf>=2.0
        :return:
        '''
        questions = ["What is your age?"]
        responses = ["I am 20 years old.", "good morning"]
        response_contexts = ["I will be 21 next year.", "great day."]

        # module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"

        embed = hub.KerasLayer(module_url)

        questions_embedding = embed(questions)
        response_embedding = embed(responses)
        print(questions_embedding.shape)
        print(questions_embedding)


    def encode_sentences(self, sentences):
        def process_to_IDs_in_sparse_format(sp, sentences):
            # An utility method that processes sentences with the sentence piece processor
            # 'sp' and returns the results in tf.SparseTensor-similar format:
            # (values, indices, dense_shape)
            ids = [sp.EncodeAsIds(x) for x in sentences]
            max_len = max(len(x) for x in ids)
            dense_shape = (len(ids), max_len)
            values = [item for sublist in ids for item in sublist]
            indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
            return (values, indices, dense_shape)


        input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        embeddings = self.embed(
            inputs=dict(
                values=input_placeholder.values,
                indices=input_placeholder.indices,
                dense_shape=input_placeholder.dense_shape))

        values, indices, dense_shape = process_to_IDs_in_sparse_format(self.sp, sentences)

        sentences_embedding = self.session.run(embeddings,
                                                feed_dict={input_placeholder.values: values,
                                                          input_placeholder.indices: indices,
                                                          input_placeholder.dense_shape: dense_shape})

        return sentences_embedding


    def encode_tf(self, sentence_ref, sentence_cands):

        ref_embeddings = self.session.run(self.embed(sentence_ref))
        cands_embeddings = self.session.run(self.embed(sentence_cands))
        similarities = self.session.run(tf.matmul(ref_embeddings, cands_embeddings, transpose_b=True))
        return similarities


if __name__ == "__main__":
    use = UniversalSentenceEncoder()
    # s1 = use.encode_tf(["My shirt is blue."], ["the shirt is good.", "the code is not good."])
    # print(s1)
    # s2 = use.encode_tf(["My shirt is blue."], ["the shirt is good.", "the code is not good."])
    # print(s2)
    #s3 = use.encode_tf(["My shirt is blue."], ["the shirt is good.", "the code is not good."])
    #print(s3)

    t1 = time.time()
    # sentences = [
    #     "The quick brown fox jumps over the lazy dog.",
    #     "I am a sentence for which I would like to get its embedding",
    #     "Although many ternary operators are possible, the conditional operator is so common, and other ternary operators so rare,"
    #     " that the conditional operator is commonly referred to as the ternary operator."]
    # sentences = 100 * sentences

    sentences = ["I have to book for the hotel.", "There are a lot of good places around the city",
                 "The atoms are created of proton and electrons", "The nature of dark matter is unknown"]

    use.encode_sentences(sentences)
    t2 = time.time()
    print(t2 - t1)