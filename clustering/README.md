# K-means clustering:

## To learn more about kmeans refer this link:https://www.analyticsvidhya.com/blog/2021/06/k-means-clustering-and-transfer-learning-for-image-classification/

## To perform clustering using kmeans, run this command:

```
python3 kmeans.py --runs_folder <path to folder containing the extracted features> --max_clusters <number of clusters> --source_folder <path to folder containing images>
```

Example:
```bash
python3 kmeans.py --runs_folder /home/wi/yolo_final/runs4 --max_clusters 2 --source_folder /home/wi/yolo_final/plastic
```

# Dbscan clustering:

## To learn more about dbscan refer this link:https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html

## To perform clustering using dbscan, run this command:

```
python3 dbscan.py 
```