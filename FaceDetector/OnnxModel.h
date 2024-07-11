#pragma once
#include "core/session/onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
typedef ONNXTensorElementDataType DataType;
struct TensorInfo
{
	std::string mName;
	std::vector<int64_t> mShape;
	DataType mType;
};
struct MemoryMan
{
	MemoryMan(const TensorInfo& tenInfo, Ort::Allocator* alloc, const Ort::MemoryInfo& mi);
	~MemoryMan();
	void* mMemory = nullptr;
	Ort::Value* mBound = nullptr;
	TensorInfo mTensorInfo;
	size_t mSizeWithByte = 0;
	size_t mSize = 1;
private:
	Ort::Allocator* mMalloc = nullptr;
};
struct Yolov8FinalResult
{
	int classId = 0;
	cv::Rect box = { 0,0,0,0 };
	cv::Mat* Mat = nullptr;
};
typedef std::list<Yolov8FinalResult> OutputResultFormat;
class OnnxModel
{
public:
	OnnxModel();
	virtual ~OnnxModel();
	bool init(const std::string& modelpath, const int& threads,
		const bool& useXnnpack);
	
private:
	bool processOrtStatus(OrtStatusPtr status_expr);
#ifdef USE_GPU
	bool initCudaOpentions(const Ort::SessionOptions& option);
#endif // USE_GPU

protected:
	Ort::Session* m_session = nullptr;
	
	Ort::IoBinding* m_binding = nullptr;
	std::vector<MemoryMan*>m_inputMemoryVect;
	std::vector<MemoryMan*>m_outputMemoryVect;
private:
	Ort::Env m_onnxEnv;
	Ort::Allocator* m_ortAllocator = nullptr;
	
};

