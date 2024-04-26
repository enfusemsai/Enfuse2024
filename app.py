import gradio as gr
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np

model = SentenceTransformer('msmarco-distilbert-base-tas-b-final')

def softmax(z):
    assert len(z.shape) == 2

    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def text_list(queries, document):
    return sum(queries, []), document  # inputs.split("--")


def score_cos_sim(art1, art2):
    scores = util.cos_sim(art1, art2)[0]
    return scores


def score_inference(queries, document):
    score = dict()
    queries = sum(queries, [])
    queries_encode = [model.encode(text) for text in queries]
    document_encode = model.encode(document)

    for i, query in enumerate(queries_encode):
        score["document_" + str(i + 1)] = score_cos_sim(query, document_encode)

    cosine_scores = np.array(
        [[i[0].numpy().tolist() for i in list(score.values())]])  # np.array([list(score.values())])

    return dict(zip(list(score.keys()), list(softmax(cosine_scores)[0])))


text_input = gr.Textbox(lines=4, placeholder="Documents Here...")

interface = gr.Interface(fn=score_inference, inputs=[gr.List(row_count=3, datatype='str'), text_input], outputs="label",
                         examples=[[[[
                                         """I have 3.5+ years of work experience and was working as a data scientist with 3 different organizations. I was responsible for using predictive modelling, data processing, and data mining algorithms to solve challenging business problems.\nMy technology stack includes but not limited to, are python, machine learning, deep learning, time-series, web scraping, flask, FastAPI, snowflake SQL servers, deploying production based servers, keras, TensorFlow, hugging face, Big Data and Data Warehouses. In my career, my growth has been exponential, and I developed interpersonal skills, now I know how to handle a project end to end.\nMy area of interests are applied machine learning, deep neural network, time series and everything around NLP in the field of ecommerce and consumer internet. My research focus is on information retrieval involving neuroscience and deep reinforcement learning.\nI like to listen to a lot of learning courses and read research papers involving deep learning. In my spare time I like to keep up with the news, read blogs on medium and watch a few sci-fi films."""],
                                     [
                                         """Snehil started his entrepreneurial journey 14 years ago with the launch of a social networking site along with music and video streaming portals back in 2006, while he was still in school. In 2011 while pursuing engineering in Computer Science, he joined Letsbuy, an e-commerce startup, where he developed and launched their mobile app and site while mobile-commerce was still in its nascent stage in India. Letsbuy was later acquired by Flipkart in 2012.Snehil also co-founded Findyahan, a services marketplace, which was eventually acquired in 2016 by Zimmber. Snehil joined Zimmber as Vice President of Product & Marketing. Zimmber was later acquired by Quikr."""],
                                     [
                                         """I have over 7 years of combined experience in the fields of data science and machine learning. I've led many data science projects in a wide array of industries. I mainly program in Python using its popular data science libraries.For deep learning, my go to framework is PyTorch. I’ve also worked a significant amount with relational databases and cloud environments.Worked on diverse array of projects where I used my machine learning expertise to build and advise external clients on how to move forward with machine learning projects. I also advised on how to best collect and structure data.Other than work, I write a significant amount with regards to AI. I’ve published several deep learning tutorials,focusing on the PyTorch framework. My articles are published on Medium under the publication A Coder’s Guide to AI."""]],
                                    "B.Tech / M.Tech degree in Computer Science from a premiere institute.\nShould have 1 - 5 years of experience in designing, developing and deploying software, preferably Statistical and Machine Learning models.\nAbility to work independently with strong problem solving skills.\nShould have excellent knowledge in fundamentals of Machine Learning and Artificial Intelligence, especially in Regression, Forecasting and Optimization.\nShould have excellent foundational knowledge in Probability, Statistics and Operations Research/Optimization techniques.\nShould have hands on experience thorugh ML Lifecycle from EDA to model deployment.\nShould have hands on experience data analysis tools like Jupyter, and packages like Numpy, Pandas, Matplotlib.\nShould be hands-on in writing code that is reliable, maintainable, secure, performance optimized.\nShould have good knowledge in Cloud Platforms and Service oriented architecture and design"]],
                         description='Enter text in the fields below and add tabs for more then 3 queries else try out the example given below by clicking it')

interface.launch()