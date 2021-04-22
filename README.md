	Intro:-
        Research paper categorization using machine learning and NLP:-

        -An interesting application of text classification is to categorize research papers Usually, a researcher has to consult with their supervisors and search extensively to find 
        To define a research idea.
        This problem can be solved to some extent using machine learning techniques e.g. classification algorithms like SVM, Naïve Bayes, etc.
        Thus, the objective of this tutorial is to provide hands-on experience on how to perform text classification using  Research paper dataset. 
        We will learn how to apply various classification algorithms to categorize research papers by Research along with feature selection and dimensionality reduction methods using a popular scikit-learn library in Python.

        Data Description:-
        Using Natural Language Processing (NLP) on NIPS papers to uncover the trendiest topics in machine learning research over the years.
        The NIPS conference (Neural Information Processing Systems) is one of the most prestigious yearly events in the machine learning community. At each NIPS conference, a large number of research papers are published. Over 20 PDF files were automatically downloaded and processed to obtain a dataset on various machine learning techniques. These NIPS papers are stored in datasets/papers.csv. The CSV file contains information on the different NIPS papers that were published from 1987 until 2021(34 years!). These papers discuss a wide variety of topics in machine learning, from neural networks 
        Since there are a lot of sections and section specific data, PDF documents contain varied information .
        

        Problem Use Case
        1-Collaborating Filtering
        2-Document Classification
        3-Creating own Embedding for NLP domain applications
        4-Academic researchers often need to face with a large collection of research papers in the literature. This problem may be even worse for postgraduate students who are new to a field and may not know where to start
        
        This could be data related but not restricted to:
        1-LOCAL RELATION NETWORKS FOR IMAGE RECOGNITION
        2-GENERAL PERCEPTION WITH ITERATIVE ATTENTION
        3-STYLE CLIP: TEXT-DRIVEN MANIPULATION OF STYLEGAN IMAGERY
        4-SHAPE AND MATERIAL CAPTURE AT HOME
        5-DEEP CONTEXTUALIZED WORD REPRESENTATIONS
        6-NEURAL MACHINE TRANSLATION OF RARE WORDS WITH SUB WORD UNITS
        

        Python Libraries Used:-
        1-Pandas
        2-Numpy
        3-os
        4-Pickle
        5-re (Regular expressions)
        6-nltk
        7-Matplotlib
        8-wordcloud
        9-CountVectorizer
        10-warnings
        11-Data Structures used:-


        Data Ingestion:-

        Input data is in the form of unstructured text-based pdf documents.

        As such, I used PyPDF2 library to process these PDF files into text readable by Python.

         I decided to extract data from abstract  for each pdf document.

        Text data from each document was stored as a single row in a pandas data frame and subsequently output to excel.

        Data Pre-processing:-

            Feature Creation — Text Length
            Feature Binning — Text length
            Transformation — Feature binning.

        Feature engineering:-

        Vectorizing the tokenised text before feeding to ML model.

        Tested and implemented following vectorisation algorithms.

            Count Vectorizer (Count)
            N-gram Vectorizer
            Tf-Idf Vectorizer (Feature weighing)

        Results:-
        we have developed an online catalog of research papers where the papers have been automatically categorized by a topic model. The catalog contains 20 papers from the proceedings of two artificial intelligence conferences from 2021 to 1992. Rather than the commonly used Latent Dirichlet Allocation, we use a recently proposed method called hierarchical latent tree analysis for topic modeling.


        Conclusion/Final Thoughts:-

        Text data, by its very nature is unstructured and highly complex to analyse.

        Resources:-
        https://medium.com/time-to-work/datacamp-project-the-hottest-topics-in-machine-learning-bcdea75abef3
        https://www.kaggle.com/arpitdw/the-hottest-topics-in-machine-learning?select=papers.csv
        
