Naive Bayes with Scikit Learn
=============================
In this Notebook Gaussian Naive Bayes is used on wisconsin cancer dataset to classify if it is Malignant or Benign

In the following pandas is used for showing our dataset. We are going to download the csv file and load it to pandas dataframe

.. code:: ipython3

    import pandas as pd

.. code:: ipython3

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

The segment of df is given below. Which shows Upper 5 rows of our data

.. code:: ipython3

    df.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>0</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>...</th>
          <th>22</th>
          <th>23</th>
          <th>24</th>
          <th>25</th>
          <th>26</th>
          <th>27</th>
          <th>28</th>
          <th>29</th>
          <th>30</th>
          <th>31</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>842302</td>
          <td>M</td>
          <td>17.99</td>
          <td>10.38</td>
          <td>122.80</td>
          <td>1001.0</td>
          <td>0.11840</td>
          <td>0.27760</td>
          <td>0.3001</td>
          <td>0.14710</td>
          <td>...</td>
          <td>25.38</td>
          <td>17.33</td>
          <td>184.60</td>
          <td>2019.0</td>
          <td>0.1622</td>
          <td>0.6656</td>
          <td>0.7119</td>
          <td>0.2654</td>
          <td>0.4601</td>
          <td>0.11890</td>
        </tr>
        <tr>
          <th>1</th>
          <td>842517</td>
          <td>M</td>
          <td>20.57</td>
          <td>17.77</td>
          <td>132.90</td>
          <td>1326.0</td>
          <td>0.08474</td>
          <td>0.07864</td>
          <td>0.0869</td>
          <td>0.07017</td>
          <td>...</td>
          <td>24.99</td>
          <td>23.41</td>
          <td>158.80</td>
          <td>1956.0</td>
          <td>0.1238</td>
          <td>0.1866</td>
          <td>0.2416</td>
          <td>0.1860</td>
          <td>0.2750</td>
          <td>0.08902</td>
        </tr>
        <tr>
          <th>2</th>
          <td>84300903</td>
          <td>M</td>
          <td>19.69</td>
          <td>21.25</td>
          <td>130.00</td>
          <td>1203.0</td>
          <td>0.10960</td>
          <td>0.15990</td>
          <td>0.1974</td>
          <td>0.12790</td>
          <td>...</td>
          <td>23.57</td>
          <td>25.53</td>
          <td>152.50</td>
          <td>1709.0</td>
          <td>0.1444</td>
          <td>0.4245</td>
          <td>0.4504</td>
          <td>0.2430</td>
          <td>0.3613</td>
          <td>0.08758</td>
        </tr>
        <tr>
          <th>3</th>
          <td>84348301</td>
          <td>M</td>
          <td>11.42</td>
          <td>20.38</td>
          <td>77.58</td>
          <td>386.1</td>
          <td>0.14250</td>
          <td>0.28390</td>
          <td>0.2414</td>
          <td>0.10520</td>
          <td>...</td>
          <td>14.91</td>
          <td>26.50</td>
          <td>98.87</td>
          <td>567.7</td>
          <td>0.2098</td>
          <td>0.8663</td>
          <td>0.6869</td>
          <td>0.2575</td>
          <td>0.6638</td>
          <td>0.17300</td>
        </tr>
        <tr>
          <th>4</th>
          <td>84358402</td>
          <td>M</td>
          <td>20.29</td>
          <td>14.34</td>
          <td>135.10</td>
          <td>1297.0</td>
          <td>0.10030</td>
          <td>0.13280</td>
          <td>0.1980</td>
          <td>0.10430</td>
          <td>...</td>
          <td>22.54</td>
          <td>16.67</td>
          <td>152.20</td>
          <td>1575.0</td>
          <td>0.1374</td>
          <td>0.2050</td>
          <td>0.4000</td>
          <td>0.1625</td>
          <td>0.2364</td>
          <td>0.07678</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 32 columns</p>
    </div>



So from the df.head() function you can see that column 1 contains the label which denotes if it is benign or malignant cancer. From column 2 to column 31 contains the features.

So we are going to prepare our training set in the following lines. X will contain featuresets and y will contain labels of each row

.. code:: ipython3

    X = df.loc[:, 2:].values

.. code:: ipython3

    y = df.loc[:, 1].values

After that we have to encode labels of y for our training purpose

Before encoding

.. code:: ipython3

    y




.. parsed-literal::

    array(['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
           'M', 'M', 'M', 'M', 'M', 'M', 'B', 'B', 'B', 'M', 'M', 'M', 'M',
           'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'B', 'M',
           'M', 'M', 'M', 'M', 'M', 'M', 'M', 'B', 'M', 'B', 'B', 'B', 'B',
           'B', 'M', 'M', 'B', 'M', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'M',
           'M', 'B', 'B', 'B', 'B', 'M', 'B', 'M', 'M', 'B', 'M', 'B', 'M',
           'M', 'B', 'B', 'B', 'M', 'M', 'B', 'M', 'M', 'M', 'B', 'B', 'B',
           'M', 'B', 'B', 'M', 'M', 'B', 'B', 'B', 'M', 'M', 'B', 'B', 'B',
           'B', 'M', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'M', 'M', 'M', 'B', 'M', 'M', 'B', 'B', 'B', 'M', 'M', 'B', 'M',
           'B', 'M', 'M', 'B', 'M', 'M', 'B', 'B', 'M', 'B', 'B', 'M', 'B',
           'B', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'M', 'B', 'B', 'B', 'B', 'M', 'M', 'B', 'M', 'B', 'B', 'M', 'M',
           'B', 'B', 'M', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'M', 'M',
           'M', 'B', 'M', 'B', 'M', 'B', 'B', 'B', 'M', 'B', 'B', 'M', 'M',
           'B', 'M', 'M', 'M', 'M', 'B', 'M', 'M', 'M', 'B', 'M', 'B', 'M',
           'B', 'B', 'M', 'B', 'M', 'M', 'M', 'M', 'B', 'B', 'M', 'M', 'B',
           'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'M', 'M', 'B', 'B', 'M',
           'B', 'B', 'M', 'M', 'B', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'B',
           'B', 'B', 'B', 'M', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
           'M', 'M', 'M', 'M', 'M', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'M',
           'B', 'M', 'B', 'B', 'M', 'B', 'B', 'M', 'B', 'M', 'M', 'B', 'B',
           'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'M', 'B',
           'B', 'M', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'B', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'M', 'B', 'M', 'B',
           'B', 'B', 'B', 'M', 'M', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'M',
           'B', 'M', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'M', 'M', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'B', 'M', 'M', 'B', 'M', 'M', 'M', 'B', 'M', 'M', 'B', 'B', 'B',
           'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'M',
           'B', 'B', 'M', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'M', 'B', 'B',
           'B', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'M', 'B',
           'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'B', 'M', 'B', 'M', 'M', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'M',
           'B', 'B', 'M', 'B', 'M', 'B', 'B', 'M', 'B', 'M', 'B', 'B', 'B',
           'B', 'B', 'B', 'B', 'B', 'M', 'M', 'B', 'B', 'B', 'B', 'B', 'B',
           'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'M', 'B',
           'B', 'B', 'B', 'B', 'B', 'B', 'M', 'B', 'M', 'B', 'B', 'M', 'B',
           'B', 'B', 'B', 'B', 'M', 'M', 'B', 'M', 'B', 'M', 'B', 'B', 'B',
           'B', 'B', 'M', 'B', 'B', 'M', 'B', 'M', 'B', 'M', 'M', 'B', 'B',
           'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'M', 'B', 'M', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
           'B', 'B', 'B', 'M', 'M', 'M', 'M', 'M', 'M', 'B'], dtype=object)



.. code:: ipython3

    from sklearn.preprocessing import LabelEncoder

.. code:: ipython3

    le = LabelEncoder()

.. code:: ipython3

    y = le.fit_transform(y)

After encoding M = 1 and B = 0.

.. code:: ipython3

    y




.. parsed-literal::

    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
           0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
           0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
           0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
           1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0,
           1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
           0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
           0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
           1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
           0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
           0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
           0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], dtype=int64)



In the following segment i'm going to split the dataset into Training and Test set with 80:20 ratio

.. code:: ipython3

    from sklearn.model_selection import train_test_split

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)

And don't forget to standardize your featuresets

.. code:: ipython3

    from sklearn.preprocessing import StandardScaler

.. code:: ipython3

    stdsc = StandardScaler()

.. code:: ipython3

    X_train_std = stdsc.fit_transform(X_train)

.. code:: ipython3

    X_test_std = stdsc.transform(X_test)

So here we are. Time for fitting our estimator with the training data.

.. code:: ipython3

    from sklearn.naive_bayes import GaussianNB

.. code:: ipython3

    clf = GaussianNB()

.. code:: ipython3

    clf.fit(X_train_std, y_train)




.. parsed-literal::

    GaussianNB(priors=None)



.. code:: ipython3

    y_pred = clf.predict(X_test_std)

y\_pred holds the predicted label of your test set.

Finally time to see the accuracy of our estimator.

.. code:: ipython3

    from sklearn.metrics import accuracy_score

.. code:: ipython3

    accuracy_score(y_true=y_test, y_pred=y_pred)




.. parsed-literal::

    0.94736842105263153



Voila!!! 94%. So end of the story.
