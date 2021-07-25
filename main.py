import time
import numpy as np
from annoy import AnnoyIndex

def ndcg(recs, gt):
    import math
    Q, S = 0.0, 0.0
    for u, vs in gt.items():
        rec = recs.get(u, [])
        if not rec:
            continue

        idcg = sum([1.0 / math.log(i + 2, 2) for i in range(len(vs))])
        dcg = 0.0
        for i, r in enumerate(rec):
            if r not in vs:
                continue
            rank = i + 1
            dcg += 1.0 / math.log(rank + 1, 2)
        ndcg = dcg / idcg
        S += ndcg
        Q += 1
    return S / Q

def inv_sigmoid(value):
    return np.log(value/(1-value))

def sample_from_gmm(f, cluster_num, num_sample):
    mus = [ np.random.rand(f) * (i*10) for i in range(1, cluster_num+1) ]
    covs = [ np.eye(f) for _ in range(cluster_num) ]

    # mus = [np.array([0, 1]), np.array([7, 10]), np.array([-3, -3])]
    # covs = [np.array([[1, 2], [2, 1]]), np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 0.3]])]

    pis = np.random.rand(cluster_num)
    pis = pis /np.sum(pis)
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]
    assert np.isclose(acc_pis[-1], 1)

    n = num_sample
    samples = []

    for i in range(n):
        # sample uniform
        r = np.random.uniform(0, 1)
        # select gaussian
        k = 0
        for i, threshold in enumerate(acc_pis):
            if r < threshold:
                k = i
                break

        selected_mu = mus[k]
        selected_cov = covs[k]

        # sample from selected gaussian
        lambda_, gamma_ = np.linalg.eig(selected_cov)

        dimensions = len(lambda_)
        # sampling from normal distribution
        y_s = np.random.uniform(0, 1, size=(dimensions * 1, 3))
        x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
        # transforming into multivariate distribution
        x_multi = (x_normal * lambda_) @ gamma_ + selected_mu
        samples.append(x_multi.tolist()[0])

    return samples

cluster_num = 10
for n_tree in [100]:
    print('n_tree', n_tree)
    start = 1000
    for n in [start]+[ start * (x * 10) for x in range(1,10,3)] + [ start * (x * 100) for x in range(1,10,3)] + [start * start]:
        n_user = n_item = n

        f = 40

        U = sample_from_gmm(f, cluster_num, n_user)
        V = sample_from_gmm(f, cluster_num, n_user)

        U = np.array(U)
        V = np.array(V).T
        selected_user = range(n_user)
        selected_user = np.random.choice(selected_user,1)[0]
        s = time.time()
        t = AnnoyIndex(f, 'dot')
        vector = np.concatenate([V.T])
        for i in range(n_item):
            t.add_item(i, vector[i])

        t.build(n_tree)

        e = time.time()
        #print(e - s)

        s = time.time()
        res, dis = t.get_nns_by_vector(U[selected_user],100, include_distances=True)
        res = {'a':list(res)}
        #print(res)
        e = time.time()
        nns_time = e - s
        #print(nns_time)

        s = time.time()
        real_res = np.argsort(U[selected_user] @ V)[::-1][:100]
        #print(real_res)
        dot_dis = (U[selected_user] @ V)[real_res]
        real_res = {'a': list(real_res)}
        e = time.time()
        dot_time = e- s
        #print(dot_time)

        acc = sum([1 for x in res if x in real_res])/len(real_res)
        print('# of user, item', n )
        print('ndcg by base', ndcg(res, real_res))
        print('dot consumed time', dot_time)
        print('nns consumed time', nns_time)
        print('speed', dot_time / nns_time,'x vs base')
        print('nns distance', 'min',min(dis),'max',max(dis))
        print('dot distance', 'min',min(dot_dis),'max',max(dot_dis))
        print('###################')
        print()
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
