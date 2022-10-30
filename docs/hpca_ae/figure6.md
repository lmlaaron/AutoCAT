# Figure 6


```
$ cd ${GIT_ROOT}/src/stealthy_streamline
```

## use the script to build all the code

```
$ build.sh
```

## measure the LRU attack bandwidth and error rate

```
$ cd ${GIT_ROOT}/src/stealthy_streamline
$ cd covert_channel_LRU_1thread_8way
$ bash run.sh
$ cd test
$ ../../process_error_rate_1thread/Error_rate_HW_LRU
```

This step will generate ```Error_rate_s_i.txt``` files. The last line of the file shows the error rate）
Currently, the threshold in process_error_rate_1thread is set for the fukushima machine. If we want to test on more machines, the threshold should be modified.


## measure the StealthyStreamline attack bandwidth and error rate

```
$ cd ${GIT_ROOT}/src/stealthy_streamline
$ cd covert_channel_stream_1thread_2bits_8way
$ bash run.sh
$ cd test
$ ../../process_error_rate_1thread/Error_rate_HW_LRU 
```

## plot the figure

```
$ cd ${GIT_ROOT}/src/stealthy_streamline/plot
$ python plot_error_rate.py
```
which will generate ```stealthy_streamline_error.png``` in the current directory. 

Which looks like

![](../../fig/stealhy_streamline_error.png)

## running Spectre attack with StreathyStreamline

```
$ cd ${GIT_ROOT}/src/stealthy_streamline
$ ./LRUspectre 
```
