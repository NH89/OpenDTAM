/*
 *  Track_align.cpp
 *  
 *
 *  Created by Paul Foster on 6/4/14.
 *  
 *
 */


#include "../include/Track.hpp"
//#include "Align_part.cpp"  // included below
#include "../include/tictoc.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>


/*
 *  Align_part.cpp
 *  
 *
 *
 */
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include "utils/utils.hpp"
#include "../include/graphics.hpp"
#include "../include/Track.hpp"
#include "stdio.h"


//debug
#define QUIET_DTAM 1
#include "../include/quiet.hpp"


const static float FAIL_FRACTION=0.30;

enum alignment_modes{CV_DTAM_REV,CV_DTAM_FWD,CV_DTAM_ESM};
const double small0=.1;//~6deg, not trivial, but hopefully enough to make the translation matter

static void getGradient(const Mat& image,Mat & grad);




//Mat& reprojectWithDepth(const Mat& T,
//                        const Mat& d,
//                        const Mat& I,
//                        const Mat& cameraMatrix,//Mat_<double>
//                        const Mat& _p,          //Mat_<double>
//                        int mode){
//
//}

static Mat paramsToProjection(const Mat & p,const Mat& _cameraMatrix){
    //Build the base transform
    assert(p.type()==CV_64FC1);
    Mat dR=rodrigues(p.colRange(Range(0,3)));
    Mat dT=p.colRange(Range(3,6)).t();
    Mat dA;
    hconcat(dR,dT,dA);
    dA=make4x4(dA);
    Mat cameraMatrix=make4x4(_cameraMatrix);
    assert(cameraMatrix.type()==CV_64FC1);
    Mat proj=cameraMatrix*dA*cameraMatrix.inv();
//     cout<<"p: "<<"\n"<< p<< endl;
//     cout<<"Proj: "<<"\n"<< proj<< endl;
    //The column swap
    Mat tmp=proj.colRange(2,4).clone();
    tmp.col(1).copyTo(proj.col(2));
    tmp.col(0).copyTo(proj.col(3));
    //The row drop
    proj=proj.rowRange(0,3);
    return proj;
}

static Mat&  makeGray(Mat& image){
    if (image.channels()!=1) {
        cvtColor(image, image, CV_BGR2GRAY);
    }
    return image;
}

static void getGradient(const Mat& image,Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    Mat grad_x(image.rows,image.cols,CV_32FC1,grad.row(0).data);
    Scharr( gray, grad_x, CV_32FC1, 1, 0, 1.0/26.0, 0, BORDER_REPLICATE );
    Mat grad_y(image.rows,image.cols,CV_32FC1,grad.row(1).data);
    Scharr( gray, grad_y, CV_32FC1, 0, 1, 1.0/26.0, 0, BORDER_REPLICATE);
}

static void getGradient_8(const Mat& image,Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    Mat grad_x(image.rows,image.cols,CV_32FC1,grad.row(0).data);
    Scharr( gray, grad_x, CV_32FC1, 1, 0, 1.0/26.0, 0, BORDER_REPLICATE );
    Mat grad_y(image.rows,image.cols,CV_32FC1,grad.row(1).data);
    Scharr( gray, grad_y, CV_32FC1, 0, 1, 1.0/26.0, 0, BORDER_REPLICATE);
}

static void getGradientInterleave(const Mat& image,Mat & grad){
    //Image gradients for alignment
    //Note that these gradients have theoretical problems under the sudden 
    // changes model of images. It might be wise to blur the images before 
    // alignment, to avoid sudden changes, but that makes occlusion more 
    // problematic.
    grad.create(2,image.rows*image.cols,CV_32FC1);
    Mat gray;
    if (image.type()==CV_32FC1) {
        gray=image;
    }else {
        cvtColor(image, gray, CV_BGR2GRAY);
        gray.convertTo(gray,CV_32FC1);
    }
    Mat gradX(image.rows,image.cols,CV_32FC1);
    Scharr( gray, gradX, CV_32FC1, 1, 0, 1.0/26.0, 0, BORDER_REPLICATE );
    Mat gradY(image.rows,image.cols,CV_32FC1);
    Scharr( gray, gradY, CV_32FC1, 0, 1, 1.0/26.0, 0, BORDER_REPLICATE);
    Mat src [2]={gradY,gradX};
    merge(src,2,grad);
}

static void Mask(const Mat& in,const Mat& m,Mat& out){
    Mat tmp;
    
    m.convertTo(tmp,in.type());
    out=out.mul(tmp/255);
}

bool Track::align_level_largedef_gray_forward(const Mat& T,//Total Mem cost ~185 load/stores of image
                          const Mat& d,
                          const Mat& _I,
                          const Mat& cameraMatrix,//Mat_<double>
                          const Mat& _p,                //Mat_<double>
                          int mode,
                          float threshold,
                          int numParams
                                      )
{

    int r=_I.rows;
    int rows=r;
    int c=_I.cols;
    int cols=c;
    const float small=small0;
    //Build the in map (Mem cost 3 layer store:3)
    Mat_<Vec3f> idMap3;
    {
        idMap3.create(r,c);//[rows][cols][3]
        float* id3=(float*) (idMap3.data);
        float* dp=(float*) (d.data);
        int offset=0;
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++,offset++){
                id3[offset*3+0]=j;
                id3[offset*3+1]=i;
                id3[offset*3+2]=dp[offset];
            }
        }
    }
    
    //Build the unincremented transform: (Mem cost 2 layer store,3 load :5)
    Mat baseMap(rows,cols,CV_32FC2);
    {
        Mat tmp=_p.clone();
        Mat baseProj=paramsToProjection(_p,cameraMatrix);
        perspectiveTransform(idMap3,baseMap,baseProj);
        assert(baseMap.type()==CV_32FC2);
    }
    
    
    // reproject the gradient and image at the same time (Mem cost >= 24)
    Mat gradI;
    Mat I(r,c,CV_32FC1);
    {
        getGradient(_I,gradI); //(Mem cost: min 2 load, 2 store :4)
        Mat toMerge[3]={_I,
                        Mat(r,c,CV_32FC1,(float*)gradI.data),
                        Mat(r,c,CV_32FC1,((float*)gradI.data)+r*c)};
        Mat packed;
        merge(toMerge,3,packed); //(Mem cost: min 3 load, 3 store :6)
        Mat pulledBack;
        
        remap( packed, pulledBack, baseMap,Mat(), CV_INTER_LINEAR, BORDER_CONSTANT,0.0 );//(Mem cost:?? 5load, 3 store:8)
        gradI.create(r,c,CV_32FC2);

        int from_to[] = { 0,0, 1,1, 2,2 };
        Mat src[1]={pulledBack};
        Mat dst[2]={I,gradI};
        
        mixChannels(src,1,dst,2,from_to,3);// extract the image and the resampled gradient //(Mem cost: min 3 load, 3 store :6)
        
        
        if(cv::countNonZero(I)<rows*cols*FAIL_FRACTION){//tracking failed!
            return false;
        }
        
    }
    
    // Calculate the differences and build mask for operations (Mem cost ~ 8)
    Mat fit;
    absdiff(T,I,fit);
    Mat mask=(fit<threshold)&(I>0);
    Mat err=T-I;
    
//     //debug
//     {
//         if (numParams==6){
//         pfShow("Before iteration",_I);
// //         if(I.rows==480){
// //             Mask(I,fit<.05,I);
// //             pfShow("Tracking Stabilized With Occlusion",I,0,Vec2d(0,1));
// // //             gpause();
// //         }
// //         else{
//             pfShow("After Iteration",I,0,Vec2d(0,1));
//             pfShow("To match",T);
// //         }
//         }
//     }
    
   
    
    
    
    // Build Jacobians:
    Mat Jsmall;
    Jsmall.create(numParams,rows*cols,CV_32FC1);
    int OM_OFFSET=0;//128;//offset helps keep cache from being clobbered by load/stores
    Mat outContainer;
    outContainer.create(numParams,rows*cols*2+OM_OFFSET,CV_32FC1);
    
    //TODO: Whole loop cacheable except J multiplies if CV_DTAM_REV (Mem cost whole loop 17/itr: 102)
    for (int paramNum=0; paramNum<numParams; paramNum++) {
        
        
        
        //Build the incremented transform
        assert(_p.type()==CV_64FC1);
        Mat_<double> p=_p.clone();
        p(0,paramNum)+=small;
        Mat proj=paramsToProjection(p,cameraMatrix);
        
        //get a row of dmap/dp
        Mat outMap(rows,cols,CV_32FC2,((float*)outContainer.data)+rows*cols*2*paramNum+OM_OFFSET);
        
        perspectiveTransform(idMap3,outMap,proj);//outmap=baseMap+dMap/dp*small (Mem cost 5)
        
        
        //subtract off the base to make a differential  (Mem cost 6)
        //this cannot be done below in the J loop because it would need 5 pointers
        // which is bad for cache (4-way set associative)
//         Mat t1,t2;
//         Mat tmp[2]={t1,t2};
//         split(outMap,tmp);
//         char s[500];
//         sprintf(s,"diff0:%d",paramNum);
//         pfShow(s,tmp[0]);
//         pfShow("diff1",tmp[1]);
        outMap-=baseMap;//outmap=dMap/dp*small
//         split(outMap,tmp);
//         pfShow("diff2",tmp[0]);
//         pfShow("diff3",tmp[1]);
        //multiply by appropriate gradient
        
        
        //want:J*small=dI/dMap*dMap/dp*small
        //do: Jsmall=sumChannels((outmap-idMap2).mul(merge(gradient[0],gradient[1])))
        const float * om=(const float*) (outMap.data);
        const float * bm=(const float*) (baseMap.data);
        const float * gi=(const float*) (gradI.data);
        const uchar * m = mask.data;
        float* Jp=(float*) (Jsmall.row(paramNum).data);//the row of the jacobian we're computing
        int offset = 0;
        //TODO: this loop may work faster unrolled or hand sse/avx optimized (Mem cost 6)
        //Warning this loop uses all 4 mappings in a 4 way cache!
        //Unrolling to cache line size would allow a 5th pointer to be used.
        for(float i=0;i<rows;i++){
            for(float j=0;j<cols;j++,offset++){
                float jx,jy;
                jx = om[offset*2+0]*gi[offset*2+0];
                jy = om[offset*2+1]*gi[offset*2+1];
                Jp[offset]=m[offset]?jx+jy:0;
            }
        }
        //usleep(1000000);
    }
    //now want: dp=(J'J)^-1*J'*(T-I)
    //          dp=small*(Jsmall*Jsmall')^-1*Jsmall*(T-I) since Jsmall is already transposed
    //          dp=small*Hsmallsmall^-1*Jsmall*(T-I)
    Mat Hss=Jsmall*Jsmall.t(); //Hessian (numParams^2) (Mem cost 6-36 depending on cache)
    Hss.convertTo(Hss,CV_64FC1);
    Mat Hinv=small*Hss.inv(DECOMP_SVD);  //TODO:cacheable for CV_DTAM_REV 
    Hinv.convertTo(Hinv,CV_32FC1);
    err=err.reshape(0,r*c);

    Mat dp=(Hinv*(Jsmall*err)).t();//transpose because we decided that p is row vector (Mem cost 7)
    dp.convertTo(dp,CV_64FC1);
//     cout<<"Je: \n"<<Jsmall*err<<endl;
//     cout<<"H: "<<"\n"<< Hss<< endl;
//     cout<<"Hinv: "<<"\n"<< Hinv<< endl;
//     cout<<"dp: "<<"\n"<< dp<< endl;
    
    
    //Check amount of motion
    {
        
    }
    
    //Check error
    //For the pixels that are within threshold, the average error should go down (Expensive!)
//     {
//         Mat tmp=_p.clone();
//         tmp.colRange(0,numParams)+=dp;
//         Mat newMap,newBack;
//         Mat newProj=paramsToProjection(tmp,cameraMatrix);
//         perspectiveTransform(idMap3,newMap,newProj);
//         remap( _I, newBack, newMap, Mat(), CV_INTER_LINEAR, BORDER_CONSTANT,-1.0/0.0 );
//         Mat newFit;
//         absdiff(T,newBack,newFit);
//         Mat fitDiff;
//         subtract(fit,newFit,fitDiff,mask & (newBack>0));
//         double deltaErr=sum(fitDiff)[0];
//         cout<<"Delta Err: "<< deltaErr<<endl;
//         if (deltaErr<0)
//             return false;
//     }
    _p.colRange(0,numParams)+=dp;
    return true;
}




///////////////////////  original align.part.cpp ////////////////////////



//needs: 
//Mat base
//Mat cameraMatrix
//Mat depth
//Cost cv
//cols

//Models Used:
//
// Warp is function of p, the parameters
// Model 1: Template(:)=Image(Warp(:)+dWarp(:))
// Model 2: Template(dWarp_inv(:))=Image(Warp(:))
//
// nb: the "+" means composition, i.e. Warp(:)+dWarp(:)=dWarp(Warp)
//
// 
// J1=dI/dWarp*dWarp/dp=grad(I)(Warp)*dWarp/dp
// J*dp=T-I
// (J'J)*dp=J1'I
// dp=(J'J)^-1*J'I     //A O(n*N) operation if J cacheable, else O(n^2*N) operation
//
// The first model is more correct, since Image is smooth 
// but Template is not smooth.
// However, the second allows caching the Jacobians and should 
// work if the depth is mostly smooth.
// Both models are averaged for ESM, giving a second order method
// but at the cost of being neither correct nor cachable.
// However, ESM is good for initial alignment because no depth is 
// used, so the presumed map is smooth both ways. Might go faster 
// to just cache anyway though and do more non-ESM steps in same
// amount of time.
//
// The paper is clear that it uses ESM for the initial levels of 
// the pyramid, and implies that it uses Model 1 for the full
// estimation. TODO:I would like to allow either choice to be made.
//
using namespace cv;
using namespace std;
#define LEVELS_2D 2

void createPyramid(const Mat& image,vector<Mat>& pyramid,int& levels){
    
    Mat in=image;
    if(levels==0){//auto size to end at >=15px tall (use height because shortest dim usually)
        for (float scale=1.0; scale>=15.0/image.rows; scale/=2, levels++);
    }
    assert(levels>0);
    int l2=levels-1;
    pyramid.resize(levels);
    pyramid[l2--]=in;
    
    for (float scale=0.5; l2>=0; scale/=2, l2--) {
        Mat out;

        
        resize(in,out,Size(),.5,.5,CV_INTER_AREA);
        pyramid[l2]=out;
        in=out;
    }
    
}

static void createPyramids(const Mat& base,
                           const Mat& depth,
                           const Mat& input,
                           const Mat& cameraMatrixIn,
                           vector<Mat>& basePyr,
                           vector<Mat>& depthPyr,
                           vector<Mat>& inPyr,
                           vector<Mat>& cameraMatrixPyr,
                           int& levels
){
    createPyramid(base,basePyr,levels);
    createPyramid(depth,depthPyr,levels);
    createPyramid(input,inPyr,levels);
    int l2=0;
    cameraMatrixPyr.resize(levels);
    // Figure out camera matrices for each level
    for (double scale=1.0,l2=levels-1; l2>=0; scale/=2, l2--) {
        Mat cameraMatrix=make4x4(cameraMatrixIn.clone());
        cameraMatrix(Range(0,2),Range(2,3))+=.5;
        cameraMatrix(Range(0,2),Range(0,3))*= scale;
        cameraMatrix(Range(0,2),Range(2,3))-=.5;
        cameraMatrixPyr[l2]=cameraMatrix;
    }
    
}
void Track::align(){
    align_gray(baseImage, depth, thisFrame);
};

void Track::align_gray(Mat& _base, Mat& depth, Mat& _input){
    Mat input,base,lastFrameGray;
    input=makeGray(_input);
    base=makeGray(_base);
    lastFrameGray=makeGray(lastFrame)  ;
    
    tic();
    int levels=6; // 6 levels on a 640x480 image is 20x15
    int startlevel=0;
    int endlevel=6;

    Mat p=LieSub(pose,basePose);// the Lie parameters 
    cout<<"pose: "<<p<<endl;

    vector<Mat> basePyr,depthPyr,inPyr,cameraMatrixPyr;
    createPyramids(base,depth,input,cameraMatrix,basePyr,depthPyr,inPyr,cameraMatrixPyr,levels);
    
    vector<Mat> lfPyr;
    createPyramid(lastFrameGray,lfPyr,levels);
    
    


    
    int level=startlevel;
    Mat p2d=Mat::zeros(1,6,CV_64FC1);
    for (; level<LEVELS_2D; level++){
        int iters=1;
        for(int i=0;i<iters;i++){
            //HACK: use 3d alignment with depth disabled for 2D. ESM would be much better, but I'm lazy right now.
            align_level_largedef_gray_forward(  lfPyr[level],//Total Mem cost ~185 load/stores of image
                                                depthPyr[level]*0.0,
                                                inPyr[level],
                                                cameraMatrixPyr[level],//Mat_<double>
                                                p2d,                //Mat_<double>
                                                CV_DTAM_FWD,
                                                1,
                                                3);
//             if(tocq()>.01)
//                 break;
        }
    }
    p=LieAdd(p2d,p);
//     cout<<"3D iteration:"<<endl;
    for (level=startlevel; level<levels && level<endlevel; level++){
        int iters=1;
        for(int i=0;i<iters;i++){
            float thr = (levels-level)>=2 ? .05 : .2; //more stringent matching on last two levels 
            bool improved;
            improved = align_level_largedef_gray_forward(   basePyr[level],//Total Mem cost ~185 load/stores of image
                                                            depthPyr[level],
                                                            inPyr[level],
                                                            cameraMatrixPyr[level],//Mat_<double>
                                                            p,                //Mat_<double>
                                                            CV_DTAM_FWD,
                                                            thr,
                                                            6);
            
//             if(tocq()>.5){
//                 cout<<"completed up to level: "<<level-startlevel+1<<"   iter: "<<i+1<<endl;
//                 goto loopend;//olny sactioned use of goto, the double break
//             }
//             if(!improved){
//                 break;
//             }
        }
    }
    loopend:
    
    pose=LieAdd(p,basePose);
    static int runs=0;
    //assert(runs++<2);
    toc();
    
}

// See reprojectCloud.cpp for explanation of the form 
// of the camera matrix for inverse depth reprojection.
//
// From that result, the form of the camera matrix for a 
// scaled camera is trivial:
//
// Camera Matrix scaling:
//
// [ f*s    0    (cx-.5)*s+.5    ] [  xc ] [ xp   ]
// [ 0      f*s  (cy-.5)*s+.5    ]*[  yc ]=[ yp   ]
// [ 0      0         1         0] [  wc ] [ 1   ]
// [ 0      0         0         1] [  zc ] [ 1/zc ]
// 
// 
// The equations: 
// All solvers fundamentally solve:
// J*dp=T-I
// by doing:
// (J'*J)^-1*J'*(T-I)
// The problem is we don't want to use pixels
// corresponding to occluded regions

// Track::cacheBaseDerivatives(){
//     Scharr( src_gray, g_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
//     Scharr( src_gray, g_y, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
// }




void Track::ESM(){

//     
//     //Build map differentials:
//     {
//         //build identity map
//         Mat_<Vec3f> xyin(im.rows,im.cols);
//         float* pt=(float*) (xyin.data);
//         float* d=(float*) (depth.data);
//         for(int i=0;i<im.rows;i++){
//             for(int j=0;j<im.cols;j++,pt+=3,d++){
//                 pt[0]=j;
//                 pt[1]=i;
//                 pt[2]=*d;
//             }
//         }
//         for (i=0; i<numDF; i++) {
//             //build slightly perturbed matrix
//             double small=1e-6;
//             Mat_<double> ptemp=p.clone();
//             ptemp(0,i)+=small;
//             Mat dR=rodrigues(ptemp(0,Range(0,3)));
//             Mat dT=ptemp(0,Range(3,6));
//             Mat dA;
//             hconcat(dR,dT,dA);
//             dA=make4x4(dA);
//             
//             float* pt=(float*) (xyin.data);
//             float* d=(float*) (depth.data);
//             for(int i=0;i<im.rows;i++){
//                 for(int j=0;j<im.cols;j++,pt+=3,d++){
//                     pt[0]=j;
//                     pt[1]=i;
//                     pt[2]=*d;
//                 }
//             } 
//         }
//     }
//     
//     for (int i=0;i<iters;i++) {
//         
//         p=ESMStep();
//     }
// 
    
    
    
}



// vector<double> Track::PJCRStep(const Mat& base,
//                                const Mat& depth,
//                                const Mat& input,
//                                const Mat& cameraMatrix,
//                                const vector<double> p,
//                                const vector<double>& maxStep=vector<double>()
//                                ){
//     
//     
// }
// 
// vector<double> Track::PJCFStep(const Mat& base,
//                               const Mat& depth,
//                               const Mat& input,
//                               const Mat& cameraMatrix,
//                               const vector<double>& p,
//                               const vector<double>& maxStep=vector<double>()
//                               ){
//     
//     
// }


void Track::cacheDerivatives(){
//     int r=rows;
//     int c=cols;
//     //Build the in map 
//     Mat_<Vec3f> idMap3;
//     {
//         idMap3.create(r,c);//[rows][cols][3]
//         float* id3=(float*) (idMap3.data);
//         float* dp=(float*) (depth.data);
//         int offset=0;
//         for(int i=0;i<r;i++){
//             for(int j=0;j<c;j++,offset++){
//                 id3[offset*3+0]=j;
//                 id3[offset*3+1]=i;
//                 id3[offset*3+2]=dp[offset];
//             }
//         }
//     }
//     
//     //Build the unincremented transform: (Mem cost 2 layer store,3 load :5)
//     Mat baseMap(rows,cols,CV_32FC2);
//     {
//         Mat_<double> p = Mat::zeros(1,6,CV_64FC1);
//         Mat baseProj=paramsToProjection(p,cameraMatrix);
//         Mat baseMap(r,c,CV_32FC2);
//         perspectiveTransform(idMap3,baseMap,baseProj);
//     }
//     
//     int numParams = 3;
//     for (int paramNum=0; paramNum<numParams; paramNum++) {
// 
//         //Build the incremented transform
//         Mat_<double> p = Mat::zeros(1,6,CV_64FC1);
//         p(0,paramNum)+=small;
//         Mat proj=paramsToProjection(p,cameraMatrix);
//         Mat tmp; hconcat(proj.colRange(Range(0,2)) ,proj.colRange(Range(3,4)) , tmp);//convert to 2D since only doing that for ESM
//         proj=tmp;
//         
//         //get a row of dmap/dp
//         Mat outmap=dMdp.row(paramNum);
//         
//         perspectiveTransform(idMap3,outmap,proj);//outmap=baseMap+dMap/dp*small (Mem cost 5)
//         
//         
//         //subtract off the base to make a differential  (Mem cost 6)
//         //this cannot be done below in the J loop because it would need 5 pointers
//         // which is bad for cache (4-way set associative)
//         outmap-=baseMap;//outmap=dMap/dp*small
//     }
//     //Cache the Template gradient
//     getGradientInterleave(baseImage,gradBase);
//     
//     
}

#define BSZ 16 //This should be adjusted to fit the cache size

static inline void JacobianCore(Mat& dMdp,
                                Mat& G,
                                Mat& J,
                                Mat& H,
                                int numParams)
{
    //tmp=M*G;
    //J = sumchannels(tmp);
    {
        float* tmp=(float *)malloc(sizeof(float)*BSZ*2*numParams);
        float* tp=tmp;
        const float* gp=G.ptr<float>(0);
        for(int pn=0;pn<numParams;pn++){
            const float* Mp=dMdp.ptr<float>(pn);//get a pointer to the row of the dMdp matrix
            float* Jp=J.ptr<float>(pn);
            for(int c=0;c<BSZ*2;c++){//multiply 
                tp[c]=Mp[2*c]*gp[2*c]+Mp[2*c+1]*gp[2*c+1];
            }
            for(int c=0;c<BSZ;c++){//multiply 
                Jp[c]=tp[2*c+0]+tp[2*c+1];
            }
            tp+=sizeof(float)*BSZ*2;
        }
        free(tmp);
    }
    
    H+=J*J.t();//add to the hessian accumulator TODO: this might need to be increased to hold things as doubles
}

static inline void solveJacobian(Mat& dMdp,
                          Mat& G,
                          Mat& J,
                          Mat& err,
                          Mat& p,
                          int numParams)
{
    int c=dMdp.cols;
    int nb=c/BSZ;
    Mat H(numParams,numParams,CV_64FC1);
    H=0.0;
    for (int i=0;i<nb;i++){
        int offset = i*BSZ;
        Mat _dMdp=dMdp.colRange(Range(offset,offset+BSZ));
        Mat _G=G.colRange(Range(offset,offset+BSZ));
        Mat _J=J.colRange(Range(offset,offset+BSZ));
        JacobianCore(_dMdp, _G, _J, H, numParams);
    }
    cout<<"Summed Hessian: "<<H<<endl;
    cout<<"Recalculated Hessian: "<<J*J.t()<<endl;
    // Now J has been filled out and H is complete
    p+=H.inv()*(J*err);
}







    

void ESMStep(const Mat& gradMTI,//2ch * N cols
             const Mat& dMdp,//2ch * 3 rows * N cols
             double* p,
             double* maxStep//maximum rotation
             )
{
    
    
}

// void getJacobianPiece(Mat idMap, Mat xyin, Mat p, Mat Jpiece){
//     Mat dR=rodrigues(p(0,Range(0,3)));
//     Mat dT=p(0,Range(3,6));
//     Mat dA;
//     hconcat(dR,dT,dA);
//     dA=make4x4(dA);
//     
//     Mat proj=cameraMatrix*dA*cameraMatrix.inv();
//     //The column swap
//     Mat tmp=proj.colRange(2,4).clone();
//     tmp.col(1).copyTo(proj.col(2));
//     tmp.col(0).copyTo(proj.col(3));
//     //The row drop
//     proj=proj.rowRange(0,3).clone();
//     
//     
//     Mat_<Vec2f> xyout(rows,cols);
//     perspectiveTransform(xyin,xyout,proj);//xyout is a two channel x,y map
//     
//     // convert to single channel stacked array type map
//     
//     
// }
// //Returns (J'*J)^-1
// void getHessian(Mat J,Mat H){
//     H=(J.t()*J).inv();
// }


// void calculateJacobianAtScale(){
//     rows=rows0*s;
//     cols=cols0*s;
//     I=scale(I0,s);
//     T=scale(T0,s);
//     d=scale(d0,s);
//     cameraMatrix=scaleCameraMatrix(cameraMatrix0,s);
//     
//     
//     // build the identitiy map
//     Mat_<Vec3f> idMap3(rows,cols);
//     Mat_<Vec2f> idMap2(rows,cols);
//     float* id3=(float*) (idMap3.data);
//     float* id2=(float*) (idMap2.data);
//     float* dp=(float*) (d.data);
//     int offset=0;
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++,offset++){
//             id3[offset*3+0]=j;
//             id3[offset*3+1]=i;
//             id3[offset*3+2]=dp[offset];
//             id2[offset*2+0]=j;
//             id2[offset*2+1]=i;
//         }
//     }
//     
//     Mat_<Vec2f> outmap(rows,cols);
//     vector<Mat> dMapdp[numParams]
//     for (int i=0; i<numParams; i++) {
//         //Build the incremented transform
//         Mat_<double> p_plus_dp=p.clone();
//         p_plus_dp(0,i)+=small;
//         Mat dR=rodrigues(p(0,Range(0,3)));
//         Mat dT=p(0,Range(3,6));
//         Mat dA;
//         hconcat(dR,dT,dA);
//         dA=make4x4(dA);
//         proj=cameraMatrix*dA*cameraMatrix.inv();
//         
//         perspectiveTransform(id3,outmap,proj);
//         dMapdp[i]=outmap.clone();
//         
//         
//     }
// 
//     
//     Mat dR=rodrigues(p(0,Range(0,3)));
//     Mat dT=p(0,Range(3,6));
//     Mat dA;
//     hconcat(dR,dT,dA);
//     dA=make4x4(dA);
//     
//     
//     
// }































                              
                                
