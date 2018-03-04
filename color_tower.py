#!/usr/bin/python
import pyopencl as cl
import numpy as np
import array
import time
import sys
import itertools
#Provide the list of things to pick from, and how many to pick.
#Returns a list of all permutations
def nPr(data,r):
    ret=[]
    if r==1:
        for i in data:
            ret.append([i])
    else:
        for i in range(0,len(data)):
            for i2 in nPr(data[:i]+data[i+1:],r-1):
                ret.append([data[i]]+i2)
    return ret

def printBoardConfig(s1,s2,s3,s4,s5,s6):
    nToL={1:'R',2:'G',4:'B',8:'Y',16:'P',32:'O'}
    board= [[3,2,1,6,4,5],
            [6,4,2,5,1,3],
            [4,1,5,3,6,2],
            [5,6,3,4,2,1],
            [2,3,6,1,5,4],
            [1,5,4,2,3,6]]
    perms=[None,s1,s2,s3,s4,s5,s6]
    cols=[[0 for i in range(0,6)]for i2 in range(0,6)]
    cur_row=[0]*6
    for row_num,row in enumerate(board):
        for col_num,val in enumerate(row):
            sys.stdout.write(nToL[perms[val][row_num]])
            sys.stdout.write(' ')
            cur_row[col_num]=perms[val][row_num]
            cols[row_num][col_num]=perms[val][row_num]
        print sum(cur_row)
    for i in cols:
        sys.stdout.write(str(sum(i)))
        sys.stdout.write(' ')
    print ' '
    #print nToL[s3[0]],nToL[s2[0]],nToL[s1[0]],nToL[s6[0]],nToL[s4[0]],nToL[s5[0]]

def runOpenCL():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    #               r g b y p o
    colorPerms=nPr((1,2,4,8,16,32),6)
    #length of colorPerms is now 720
    colorPerms_concat=[0]*720*6
    for idx, i in enumerate(colorPerms):
        for i2dx, i2 in enumerate(i):
            colorPerms_concat[idx*6+i2dx]=i2
    #colorPerms_concat is now a list of all the color perms, but not nested.  Just smacked together
    #373,248,000 is 720^3
    total=373248000
    mf = cl.mem_flags
    colorPerms_concat_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=memoryview(array.array("i",colorPerms_concat).tostring()))
    s5s = np.zeros(total,np.int32)
    s6s = np.zeros(total,np.int32)
    s5s_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, s5s.nbytes)
    s6s_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, s6s.nbytes)
    prg = cl.Program(ctx, """
    __kernel void check_chunk(
    __global const int *color_permutations,
    __global int *s5s, __global int *s6s)
    {
        const int y1s[6]={5,2,0,4,1,3};
        const int y2s[6]={4,0,1,5,3,2};
        const int y3s[6]={0,4,3,2,5,1};
        const int y4s[6]={2,1,5,3,0,4};
        const int y5s[6]={3,5,2,1,4,0};
        const int y6s[6]={1,3,4,0,2,5};
        int gid = get_global_id(0);
        int s1=0;
        int decode=gid;
        int s2=decode%720;
        decode/=720;
        int s3=decode%720;
        decode/=720;
        int s4=decode%720;
        int s5=0;
        int s6=0;
        bool found=false;
        for(s5=0;(s5<720)&&(!found);s5++){
            for(s6=0;(s6<720)&&(!found);s6++){
                //check rows, then columns for i=1...6
                bool maybe=true;
                for(int i=0;(i<6)&&(maybe);i++){
                    if(color_permutations[6*s1+i]+color_permutations[6*s2+i]+color_permutations[6*s3+i]+color_permutations[6*s4+i]+color_permutations[6*s5+i]+color_permutations[6*s6+i]!=63){
                        maybe=false;
                    }
                    if(color_permutations[6*s1+y1s[i]]+color_permutations[6*s2+y2s[i]]+color_permutations[6*s3+y3s[i]]+color_permutations[6*s4+y4s[i]]+color_permutations[6*s5+y5s[i]]+color_permutations[6*s6+y6s[i]]!=63){
                        maybe=false;
                    }
                }
                found=maybe;
            }
        }
        if(found){
            s5s[gid]=s5;
            s6s[gid]=s6;
        }else{
            s5s[gid]=-1;
            s6s[gid]=-1;
        }
    }
    """).build()
    start=time.time()
    print s5s.shape
    event = prg.check_chunk(queue, s5s.shape, None, colorPerms_concat_buffer, s5s_buffer, s6s_buffer)
    event.wait()
    print time.time()-start
    cl.enqueue_copy(queue,s5s,s5s_buffer)
    cl.enqueue_copy(queue,s6s,s6s_buffer)
    for idx,i in enumerate(s5s):
        if i!=-1:
            print idx,i,s6s[idx]
            decode=idx
            s2=decode%720;
            decode/=720;
            s3=decode%720;
            decode/=720;
            s4=decode%720;
            printBoardConfig(colorPerms[0],colorPerms[s2],colorPerms[s3],colorPerms[s4],colorPerms[i],colorPerms[s6s[idx]])

if __name__ == '__main__':
    colorPerms=nPr((1,2,4,8,16,32),6)
    #print len(itertools.combinations_with_replacement([1,2,4,8,16,32],6))
    printBoardConfig(colorPerms[0],colorPerms[10],colorPerms[200],colorPerms[300],colorPerms[500],colorPerms[600])
    #exit()
    runOpenCL()
