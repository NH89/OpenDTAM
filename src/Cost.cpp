#include "Cost.h"
#define COST_CPP_SUBPARTS

//debugs
#include <iostream>
#include <algorithm>
#include <math.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../utils/reproject.hpp"
#include "../../tictoc.h"
#include "../../graphics.hpp"
// #define DTAM_COST_DEBUG
//in Cost.cpp
//#include "min.part.hpp"
//#include "updateCost.part.hpp"
//#include "updateCost.part.cpp"  // pasted below
//#ifdef COST_CPP_SUBPARTS
//#include "updateCost.part.hpp"

#define COST_CPP_DATA_MIN 3
#define COST_CPP_INITIAL_WEIGHT .001



using namespace std;
static inline float fastabs(const float& foo){
    return fabs(foo);
}

void Cost::updateCostL1(const cv::Mat& image,
                                     const cv::Matx44d& currentCameraPose)
{
    imageNum++;
    cv::Mat newLo(rows,cols,CV_32FC1,1000.0);
    newLo=1000.0;
    cv::Mat newHi(rows,cols,CV_32FC1,0);
    for(int n=0; n < depth.size(); ++n){
        
       // tic();
        cv::Mat_<cv::Vec3f> plane;
        cv::Mat_<uchar> mask;
        reproject(cv::Mat_<cv::Vec3f>(image), cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);//could be as fast as .00614 using resize instead. Currently runs at .0156s, or about twice as long
        size_t end=image.rows*image.cols*image.channels();
        size_t lstep=layers;
        #ifdef DTAM_COST_DEBUG
        float* pdata;
#else
        const float* pdata;
#endif
        pdata=(float*)(plane.data);
        const float* idata=(float*)(baseImage.data);
        float* cdata=data+n;
        float* hdata=hit+n;
        //float* ldata=(float*)(newLo.data);
        float* xdata=(float*)(newHi.data);
        char*  mdata=(char*)(mask.data);
        //hdata and cdata aligned
        //pdata and idata aligned
        //size_t moff=0;
        //toc();
        //tic();
        for (size_t i=0, moff=0,coff=0,p=0;  i<end; p++, moff+=3, i+=3, coff+=lstep){//.0055 - .0060 s 

            //std::cout<<mdata[moff]<<std::endl;
            if(mdata[moff]){
                float v1=fastabs(pdata[i]-idata[i]);
                float v2=fastabs(pdata[i+1]-idata[i+1]);
                float v3=fastabs(pdata[i+2]-idata[i+2]);
                float h=hdata[coff]+1;
                float ns=cdata[coff]*(1-1/h)+(v1+v2+v3)/h;
                
                hdata[coff]=h;
                cdata[coff]=ns;
               // std::cout<<ns<<std::endl;
            }
#ifdef DTAM_COST_DEBUG
            {//debug see the cost
                pdata[i]=cdata[coff];
                pdata[i+1]=cdata[coff];
                pdata[i+2]=cdata[coff];
            }
#endif
        }
#ifdef DTAM_COST_DEBUG
        {//debug
           pfShow( "Cost Volume Slice", plane,0,cv::Vec2d(0,.5));
          // gpause();
        }
#endif
        //toc();
    }
/*//     cv::Mat loInd(rows,cols,CV_32SC1);
//     cv::Mat loVal(rows,cols,CV_32FC1);
//     minv(data,loInd,loVal);
// 
//     //         loInd.convertTo(loInd,CV_32FC1);
//     imshow( "Display window", loInd*255*255/layers);
//     //std::cout<<"Depth: "<<depth[n]<<std::endl;
//     cvWaitKey(1);*/ 
}

void Cost::updateCostL2(const cv::Mat& image,
                        const cv::Matx44d& currentCameraPose)
{
    if (image.type()==CV_32FC3){
        cv::Mat newLo(rows,cols,CV_32FC1,1000.0);
        newLo=1000.0;
        cv::Mat lInd(rows,cols,CV_32SC1);
        lInd=255;
        
        cv::Mat newHi(rows,cols,CV_32FC1,0);
        for(int n=0; n < depth.size(); ++n){
            
            cv::Mat_<cv::Vec3f> _img(image);
            cv::Mat_<cv::Vec3f> plane;
            cv::Mat_<uchar> mask;
            reproject(_img, cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);
            size_t end=image.rows*image.cols*image.channels();
            size_t lstep=layers;
            float* pdata=(float*)(plane.data);
            float* idata=(float*)(baseImage.data);
            float* cdata=data+n;
            float* hdata=hit+n;
            float* ldata=(float*)(newLo.data);
            
            float* xdata=(float*)(newHi.data);
            char*  mdata=(char*)(mask.data);
            
            //size_t moff=0;
            for (size_t i=0, moff=0,coff=0,p=0;  i<end; p++, moff+=3, i+=3, coff+=lstep){
                if (n==0){
                    ldata[p]=255.0;
                }
                //std::cout<<mdata[moff]<<std::endl;
                if(mdata[moff]){
                    float v1=pdata[i]-idata[i];
                    float v2=pdata[i+1]-idata[i+1];
                    float v3=pdata[i+2]-idata[i+2];
                    float ns=cdata[coff]+v1*v1+v2*v2+v3*v3;
                    cdata[coff]=ns;
                    
                    if(ldata[p]>(ns/hdata[coff])){
                        ldata[p]=ns/hdata[coff];
                        
                        ((int*)(lInd.data))[p]=n;
                    }
                    hdata[coff]+=1.0;
                    //std::cout<<p<<std::endl;
                }
                /*//                 {//debug see the cost
                //                 pdata[i]=cdata[coff]/hdata[coff]*10;
                //                 pdata[i+1]=cdata[coff]/hdata[coff]*10;
                //                 pdata[i+2]=cdata[coff]/hdata[coff]*10;
                //                 }*/ 
            }
            /*//             {//debug
            //             //absdiff(plane,baseImage,plane);
            //             cv::Mat tmp;
            //             lInd.convertTo(tmp,CV_32FC3,1./255.);
            //             imshow( "Display window", tmp);
            //             //std::cout<<"Depth: "<<depth[n]<<std::endl;
            //             cvWaitKey(1);
            //             }*/ 
        }
    }
    else if (image.type()==CV_32FC1){
        for(int n=0; n < depth.size(); ++n){
            cv::Mat_<float> _img(image);
            cv::Mat_<float> plane;
            cv::Mat_<uchar> mask;
            reproject(_img, cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);
            size_t end=image.rows*image.cols;
            size_t lstep=layers;
            float* pdata=(float*)(plane.data);
            float* idata=(float*)(baseImage.data);
            float* cdata=((float*)data)+n;
            float* hdata=((float*)hit)+n;
            char*  mdata=(char*)(mask.data);
            
            //size_t moff=0;
            for (size_t i=0, moff=0,coff=0;  i<end;  moff++, i++, coff+=lstep){
                //std::cout<<mdata[moff]<<std::endl;
                if(mdata[moff]){
                    float v1=pdata[i]-idata[i];
                    cdata[coff]+=v1*v1;
                    hdata[coff]++;
                    /*//                    {//debug see the cost
                    //                        pdata[i]=cdata[coff]*10;
                    //                    }
                    // std::cout<<cdata[coff]<<std::endl;*/ 
                }
            }   
        /*//            {//debug
        //                //absdiff(plane,baseImage,plane);
        //                imshow( "Display window", plane);
        //                std::cout<<"Depth: "<<depth[n]<<std::endl;
        //                cvWaitKey(0);
        //            }*/ 
        }
    }
    else{
        std::cout<<"Error, Unsupported Type!"<<std::endl;
        assert(false);
    }
}


void Cost::updateCostL1(const cv::Mat& image, const cv::Mat& R, const cv::Mat& Tr)
{
    updateCostL1(image,convertPose(R,Tr));
}

void Cost::updateCostL2(const cv::Mat& image, const cv::Mat& R, const cv::Mat& Tr)
{
    updateCostL2(image,convertPose(R,Tr));
}
//#endif
















//#include "min.part.cpp"    // pasted below


void Cost::minv(uchar* _data,cv::Mat& _minIndex,cv::Mat& _minValue){
    minv((float*) _data, _minIndex, _minValue);
}

void Cost::minv(float* _data,cv::Mat& _minIndex,cv::Mat& _minValue){
    assert(_minIndex.type()==CV_32SC1);
    int r=rows;
    int c=cols;
    int l=layers;
    _minIndex.create(rows,cols,CV_32SC1);
    _minValue.create(rows,cols,CV_32FC1);
    float* data=(float*)( _data);
    int* minIndex=(int*)(_minIndex.data);
    float* minValue=(float*)(_minValue.data);
    
    for(int i=0,id=0;i<r*c;i++){//i is offset in 2d, id is offset in 3d
        //first element is min so far
        int mi=0;
        float mv=data[id];
        id++;
        for (int il=1;il<l;il++,id++){//il is layer index
            float v=data[id];
            if(mv>v){
                mi=il;
                mv=v;
            }
        }
        minIndex[i]=mi; 
        minValue[i]=mv; 
    }
}

void Cost::maxv(float* _data,cv::Mat& _maxIndex,cv::Mat& _maxValue){
    assert(_maxIndex.type()==CV_32SC1);
    
    int r=rows;
    int c=cols;
    int l=layers;
    _maxIndex.create(rows,cols,CV_32SC1);
    _maxValue.create(rows,cols,CV_32FC1);
    float* data=(float*)( _data);
    int* maxIndex=(int*)(_maxIndex.data);
    float* maxValue=(float*)(_maxValue.data);
    
    for(int i=0,id=0;i<r*c;i++){//i is offset in 2d, id is offset in 3d
        //first element is max so far
        int mi=0;
        float mv=data[id];
        id++;
        for (int il=1;il<l;il++,id++){//il is layer index
            float v=data[id];
            if(mv<v){
                mi=il;
                mv=v;
            }
        }
        maxIndex[i]=mi; 
        maxValue[i]=mv; 
    }
}

void Cost::minmax(){
    int r=rows;
    int c=cols;
    int l=layers;
    lo.create(rows,cols,CV_32FC1);
    hi.create(rows,cols,CV_32FC1);
    float* maxValue=(float*)(hi.data);
    float* minValue=(float*)(lo.data);
    
    for(int i=0,id=0;i<r*c;i++){//i is offset in 2d, id is offset in 3d
        //first element is max so far
        float mhiv=data[id];
        float mlov=data[id];
        id++;
        for (int il=1;il<l;il++,id++){//il is layer index
            float v=data[id];
            if(mhiv<v){
                mhiv=v;
            }
            if(mlov>v){
                mlov=v;
            }
        }
        minValue[i]=mlov; 
        maxValue[i]=mhiv; 
    }
}


















#undef COST_CPP_SUBPARTS




Cost::Cost(const cv::Mat& baseImage, int layers, const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
layers(layers),
depth(generateDepths(layers)),
cameraMatrix(cameraMatrix),
pose(convertPose(R,Tr)),
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<float>::type, cv::Scalar(COST_CPP_INITIAL_WEIGHT)))//allocate enough data to hold all of the hits info in cost volume
{
    init();
}


Cost::Cost(const cv::Mat& baseImage, int layers, const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
layers(layers),
depth(generateDepths(layers)),
cameraMatrix(cameraMatrix),
pose(cameraPose),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*layers,1,cv::DataType<float>::type, cv::Scalar(COST_CPP_INITIAL_WEIGHT))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    init();
}


Cost::Cost(const cv::Mat& baseImage, const std::vector<float>& depth, const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& Tr):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
depth(depth),
layers(depth.size()),
cameraMatrix(cameraMatrix),
pose(convertPose(R,Tr)),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*depth.size(),1,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*depth.size(),1,cv::DataType<float>::type, cv::Scalar(COST_CPP_INITIAL_WEIGHT))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    init();
}


Cost::Cost(const cv::Mat& baseImage, const std::vector<float>& depth, const cv::Mat& cameraMatrix, const cv::Matx44d& cameraPose):
baseImage(baseImage),
rows(baseImage.rows),
cols(baseImage.cols),
layers(depth.size()),
depth(depth),
cameraMatrix(cameraMatrix),
pose(cameraPose),
dataContainer(cv::Mat(baseImage.rows*baseImage.cols*depth.size(),1,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),//allocate enough data to hold all of the cost volume
hitContainer(cv::Mat(baseImage.rows*baseImage.cols*depth.size(),1,cv::DataType<float>::type, cv::Scalar(COST_CPP_INITIAL_WEIGHT))),//allocate enough data to hold all of the hits info in cost volume
lo(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN))),
hi(cv::Mat(baseImage.rows,baseImage.cols,cv::DataType<float>::type, cv::Scalar(COST_CPP_DATA_MIN)))
{
    init();
}

const cv::Mat Cost::depthMap(){
    //Returns the best available depth map
    // Code should not rely on the particular mapping of true 
    // internal data to true inverse depth, as this may change.
    // Currently depth is just a constant multiple of the index, so
    // infinite depth is always represented. This is likely to change.
    if(stableDepth.data){
        return stableDepth*depthStep;
    }
    return _a*depthStep;
}


