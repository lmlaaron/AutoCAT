# run unit test
../../src/cache_simulator.py -pdc $1/config -t $1/test_nolock.txt -f $1/result_nolock.txt

if diff -u "$1/result_nolock.txt" "$1/expected_result_nolock.txt" ; then
        echo "$1 test passed! "
else
    diff -u "$1/result_nolock.txt" "$1/expected_result_nolock.txt" ; 
    echo "$1 test failed!" 
    :
fi 
