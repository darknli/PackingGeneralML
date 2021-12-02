dir_egg="./genal_ml_darkn.egg-info"
[ -d "$dir_egg" ] && rm -rf "$dir_egg" && echo "清理${dir_egg}缓存成功"

dir_dist="./dist"
[ -d "$dir_dist" ] && rm -rf "$dir_dist" && echo "清理${dir_egg}缓存成功"
	.
dir_build="./build"
[ -d "$dir_build" ] && rm -rf "*dir_build" && echo "清理${dir_egg}缓存成功"

python setup.py sdist bdist_wheel
echo "done"


