# hadoop-python

## Install local Hadoop cluster

```
git clone https://github.com/big-data-europe/docker-hadoop
cp docker-compose.yml docker-hadoop
cd docker-hadoop
sudo docker-compose up -d
docker container ls # смотрим контейнеры
```
![image](https://github.com/cr00z/hadoop-python/raw/main/images/docker.jpg)

[Статус local Hadoop cluster](http://localhost:9870)

Копируем скрипты в докер и логинимся:

```
docker cp mapper.py namenode:mapper.py
docker cp reducer.py namenode:reducer.py
# sudo chown $USER /var/run/docker.sock # если не будет прав
docker exec -it namenode /bin/bash
```

Готовим данные и заливаем в HDFS:

```
mkdir input
echo "Hello World" >input/f1.txt
echo "Hello Docker" >input/f2.txt
echo "Hello Hadoop" >input/f3.txt
echo "Hello MapReduce" >input/f4.txt

hadoop fs -mkdir -p input
hdfs dfs -put ./input/* input
```

Если namenode ругается:
> mkdir: Cannot create directory /user/root/input. Name node is in safe mode.

```hdfs dfsadmin -safemode leave```

Запускаем:

```
find / -name 'hadoop-streaming*.jar'

hadoop jar /opt/hadoop-3.2.1/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar \
-file mapper.py -mapper mapper.py \
-file reducer.py -reducer reducer.py \
-input input -output output
```

Останавливаем hadoop:

```
docker-compose down
```