profile_out=$1
shift
python -m cProfile -o $profile_out $*
#pyprof2calltree -i $profile_out -o $profile_out.calltree
#kcachegrind $profile_out.calltree
#snakeviz $profile_out
