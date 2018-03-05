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
        const int board[6][6]={
        {3,2,1,6,4,5},
        {6,4,2,5,1,3},
        {4,1,5,3,6,2},
        {5,6,3,4,2,1},
        {2,3,6,1,5,4},
        {1,5,4,2,3,6}};
        int gid = get_global_id(0);
        int s[6]={0,0,0,0,0,0};
        int decode=gid;
        s[1]=decode%720;
        decode/=720;
        s[2]=decode%720;
        decode/=720;
        s[3]=decode%720;
        //first check if there are duplicates in the 4x4 chunk that is constant for this kernel to be checking
        bool good_set=true;
        //check the rows...
        for(int row=0;row<6;row++){
            int vals[6]={0,0,0,0,0,0};
            for(int col=0;col<6;col++){
                if(board[row][col]>3){
                    continue;
                }
                if(++vals[color_permutations[6*s[board[row][col]]+row]]>1){
                    good_set=false;
                }
            }
        }
        //check the columns...
        for(int col=0;col<6;col++){
            int vals[6]={0,0,0,0,0,0};
            for(int row=0;row<6;row++){
                if(board[row][col]>3){
                    continue;
                }
                if(++vals[color_permutations[6*s[board[row][col]]+row]]>1){
                    good_set=false;
                }
            }
        }
        if(!good_set){
            s5s[gid]=-1;
            s6s[gid]=-1;
            return;
        }
        bool found=false;
        for(s[4]=0;(s[4]<720)&&(!found);s[4]++){
            for(s[5]=0;(s[5]<720)&&(!found);s[5]++){
                //check rows, then columns for i=1...6
                bool maybe=true;
                int cols[6]={0,0,0,0,0,0};
                for(int row=0;row<6 && maybe;row++){
                    int cur_row=0;
                    for(int col=0;col<6;col++){
                        int cur_val=color_permutations[6*s[board[row][col]]+row];
                        cur_row+=cur_val;
                        cols[col]+=cur_val;
                    }
                    maybe=maybe && (cur_row==63);
                }
                bool col_check=true;
                for(int i=0;i<6;i++){
                    col_check=col_check && cols[i]==63;
                }
                found=maybe&&col_check;
            }
        }
        if(found){
            s5s[gid]=s[4];
            s6s[gid]=s[5];
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
