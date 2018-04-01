wget https://fastdl.mongodb.org/osx/mongodb-osx-ssl-x86_64-3.6.3.tgz
tar -xvzf mongodb-osx-ssl-x86_64-3.6.3.tgz
mv mongodb-osx-x86_64-3.6.3 mongodb
mkdir -p data/db
./mongodb/bin/mongod --dbpath data/db > mongdb.log  2>&1 &

 
