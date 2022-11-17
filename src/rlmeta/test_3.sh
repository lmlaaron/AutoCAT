correct_num=0
wrong_num=0

for k in `seq 1 500`; do
   x='abcdefghijklmnop'
   irv=$(( $RANDOM % 8  ))
   irv2=$(( $RANDOM % 8  ))
   rv=${x:${irv}:1}
   rv2=${x:${irv2}:1}
   #str=''
   #y='aaaaaaaaijklmnop'
   #for i in `seq 0 15`; do
   #    irv3=$(( $RANDOM % 16  ))
   #    rv3=${y:${irv3}:1}
   #    str="${str} $rv3"
   #done
   #echo $str
   #./cachequery.py -l l1       k o p ${rv} i n j m a k? | #${rv2} p? n?
   r=$(./cachequery.py -l l1 k o p ${rv} i n j m a k? | tail -1 | awk '{print $12}')
   #echo $r
   if [ $r -lt 50 ]
   then
      correct_num=$((correct_num + 1))
   else
      wrong_num=$((wrong_num + 1))
   fi
   #irv=$(( $RANDOM % 8  ))
   #irv2=$(( $RANDOM % 8  ))
   #rv=${x:${irv}:1}
   #rv2=${x:${irv2}:1}
   #str=''
   #for i in `seq 0 7`; do
   #    irv3=$(( $RANDOM % 9  ))
   #    rv3=${y:${irv3}:1}
   #    str="${str} $rv3"
   #done
   #echo $str
   r=$(./cachequery.py -l l1       k o p ${rv2} i n j m   k?  | tail -1 | awk '{print $11}')
   #echo $r
   if [ $r -gt 50 ]
   then
      correct_num=$((correct_num + 1))
   else
      wrong_num=$((wrong_num + 1))
   fi
done 
   
echo "correct_num"
echo "${correct_num}"
echo "wrong_num"
echo "${wrong_num}"
