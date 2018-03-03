#!/usr/bin/python
import pyopencl as cl
import numpy as np
import array
import time
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
def runOpenCL():
    ctx = cl.create_some_context()
    #               r g b y p o
    colorPerms=nPr((1,2,3,4,5,6),6)
    #length of colorPerms is now 720
    colorPerms_concat=[0]*720*6
    for idx, i in enumerate(colorPerms):
        for i2dx, i2 in enumerate(i):
            colorPerms_concat[idx*6+i2dx]=i2
    #colorPerms_concat is now a list of all the color perms, but not nested.  Just smacked together
    #373,248,000 is 720^3
    colorPerms_concat_buffer = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=memoryview(array.array("i",colorPerms_concat).tostring()))
    s5s = np.zeros(373248000,np.int32)
    s6s = np.zeros(373248000,np.int32)
    s5s_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, s5s.nbytes)
    s6s_buffer = cl.Buffer(self.ctx, mf.WRITE_ONLY, s6s.nbytes)
    prg = cl.Program(self.ctx, """
    __kernel void check_chunk(
    __global const int *color_permutations,
    int s5s, int s6s)
    {
        const int y1s={5,2,0,4,1,3};
        const int y2s={4,0,1,5,3,2};
        const int y3s={0,4,3,2,5,1};
        const int y4s={2,1,5,3,0,4};
        const int y5s={3,5,2,1,4,0};
        const int y6s={1,3,4,0,2,5};
        int gid = get_global_id(0);
        int s1=0;
        int decocde=gid;
        int s2=decode%720;
        decode/=720;
        int s3=decode%720;
        decode/=720;
        int s4=decode%720;
        int s5=0;
        int s6=0;
        bool found=false;
        for(s5=0;s5<720;s5++){
            for(s6=0;s6<720;s6++){
                //check rows, then columns for i=1...6
                for(int i=0;i<6;i++){
                    if(color_permutations[6*s1+i]+color_permutations[6*s2+i]+color_permutations[6*s3+i]+color_permutations[6*s4+i]+color_permutations[6*s5+i]+color_permutations[6*s6+i]!=63){
                        continue;
                    }
                    if(color_permutations[6*s1+y1s[i]]+color_permutations[6*s2+y2s[i]]+color_permutations[6*s3+y3s[i]]+color_permutations[6*s4+y4s[i]]+color_permutations[6*s5+y5s[i]]+color_permutations[6*s6+y6s[i]]!=63){
                        continue;
                    }
                }
                found=true;
            }
            if(found){
                break;
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
    prg.check_chunk(queue, s5s.shape, None, colorPerms_concat_buffer, s5s_buffer, s6s_buffer)
    cl.enqueue_copy(queue,s5s,s5s_buffer)
    cl.enqueue_copy(queue,s6s,s6s_buffer)
    for idx,i in enumerate(s5s):
        if i!=-1:
            print idx,i,s6s[idx]

if __name__ == '__main__':
    runOpenCL()
