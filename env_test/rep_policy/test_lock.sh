# run unit test
../../src/cache_simulator.py -pdc $1/config -t $1/test_lock.txt -f $1/result_lock.txt

if diff -u "$1/result_lock.txt" "$1/expected_result_lock.txt" ; then
        echo "$1 test passed! "
else
    echo "$1 test failed!" 
    :
fi 
