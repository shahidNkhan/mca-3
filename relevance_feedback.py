from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    alpha = 0.75
    beta = 0.5
    rf_sim = sim # change
    query_words = {}
    word_docs = {}
    query_docs = {}

    with open('data/MED.REL', 'r') as f:
        lines = f.readlines()
    gt = [(int(l.split()[0]), int(l.split()[2])) for l in lines]

    ground_truth = {}
    for i in range(30):
        ground_truth[i] = []
    for t in gt:
        ground_truth[t[0]-1].append(t[1]-1)

    for epoch in range(3):
        sim_matrix = cosine_similarity(vec_docs, vec_queries)
        for q_ind in range(30):
            q_scores = {}
            for d_ind in range(1033):
                if sim_matrix[d_ind,q_ind]>0:
                    q_scores[d_ind] = sim_matrix[d_ind,q_ind]
            q_scores = sorted(q_scores.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
            doc_counter = 0
            good = []
            bad = []
            for d in q_scores:
                if d[0] in ground_truth[q_ind]:
                    good.append(d[0])
                else:
                    bad.append(d[0])
                doc_counter += 1
                if doc_counter == 10:
                    break
            adders = [g for g in ground_truth[q_ind] if g not in good]
            subs = bad
            addersum = 0
            for a in adders:
                addersum += vec_docs[a][:]
            # print(len(addersum))
            # print(addersum)
            addersum = addersum * alpha
            subsum = 0
            for s in subs:
                subsum += vec_docs[s][:]
            subsum = subsum * beta
            vec_queries[q_ind] = vec_queries[q_ind] + addersum - subsum
    return cosine_similarity(vec_docs, vec_queries)


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim  # change
    alpha = 0.75
    beta = 0.5
    rf_sim = sim # change
    query_words = {}
    word_docs = {}
    query_docs = {}
    tot_epochs = 3

    with open('data/MED.REL', 'r') as f:
        lines = f.readlines()
    gt = [(int(l.split()[0]), int(l.split()[2])) for l in lines]

    ground_truth = {}
    for i in range(30):
        ground_truth[i] = []
    for t in gt:
        ground_truth[t[0]-1].append(t[1]-1)

    for epoch in range(tot_epochs):
        sim_matrix = cosine_similarity(vec_docs, vec_queries)
        for q_ind in range(30):
            q_scores = {}
            for d_ind in range(1033):
                if sim_matrix[d_ind,q_ind]>0:
                    q_scores[d_ind] = sim_matrix[d_ind,q_ind]
            q_scores = sorted(q_scores.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
            doc_counter = 0
            good = []
            bad = []
            for d in q_scores:
                if d[0] in ground_truth[q_ind]:
                    good.append(d[0])
                else:
                    bad.append(d[0])
                doc_counter += 1
                if doc_counter == 10:
                    break
            adders = [g for g in ground_truth[q_ind] if g not in good]
            
            for a in adders:
                modifiers = {}
                cur_doc = vec_docs[a].todense()
                for t in range(cur_doc.shape[1]):
                    if cur_doc[0,t] > 0:
                        modifiers[t] = cur_doc[0,t]
                modifiers  = sorted(modifiers.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
                count_mod = 0
                for mod in modifiers:
                    vec_queries[q_ind,mod[0]] = vec_queries[q_ind,mod[0]] + alpha*mod[1]
                    count_mod+=1
                    if count_mod == tot_epochs:
                        break
    return cosine_similarity(vec_docs, vec_queries)