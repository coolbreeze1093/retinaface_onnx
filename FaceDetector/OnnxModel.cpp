#include "OnnxModel.h"
#include <iostream>
#include <string>
#include <codecvt>

MemoryMan::MemoryMan(const TensorInfo& tenInfo, Ort::Allocator* alloc, const Ort::MemoryInfo& mi)
	:mTensorInfo(tenInfo), mMalloc(alloc)
{
	switch (mTensorInfo.mType)
	{
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
	{
		for (auto shape : tenInfo.mShape)
		{
			mSize *= shape;
		}
		mSizeWithByte = mSize * sizeof(float);
		mMemory = mMalloc->Alloc(mSizeWithByte);
		mBound = new Ort::Value(Ort::Value::CreateTensor(mi, reinterpret_cast<float*>(mMemory),
			mSize, mTensorInfo.mShape.data(), mTensorInfo.mShape.size()));
	}
	break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
		break;
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
		break;
	default:
		break;
	}
}

MemoryMan::~MemoryMan()
{

	if (mBound)
	{
		delete mBound;
		mBound = nullptr;
	}
	if (mMalloc)
	{
		mMalloc->Free(mMemory);
		mMemory = nullptr;
	}
}

OnnxModel::OnnxModel()
{
	
}

OnnxModel::~OnnxModel()
{
	if (m_binding)
	{
		delete m_binding;
		m_binding = nullptr;
	}
	std::vector<MemoryMan*>::iterator ps = m_inputMemoryVect.begin();
	do
	{
		MemoryMan* _mm = *ps;
		ps = m_inputMemoryVect.erase(ps);
		delete _mm;
		_mm = nullptr;
	} while (ps != m_inputMemoryVect.end());

	ps = m_outputMemoryVect.begin();
	do
	{
		MemoryMan* _mm = *ps;
		ps = m_outputMemoryVect.erase(ps);
		delete _mm;
		_mm = nullptr;
	} while (ps != m_outputMemoryVect.end());

	if (m_ortAllocator)
	{
		delete m_ortAllocator;
		m_ortAllocator = nullptr;
	}
	if (m_session)
	{
		delete m_session;
		m_session = nullptr;
	}
}

bool OnnxModel::init(const std::string& modelpath, const int& threads, const bool& useXnnpack)
{
	const auto& _api = Ort::GetApi();
	// 创建会话选项
	Ort::SessionOptions session_options;
#ifdef USE_GPU
	if (!initCudaOpentions(session_options))
	{
		SIM_LOG_INFO << "init cuda error!!";
		return false;
	}
#endif // DEBUG

	// 加载模型
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	const std::wstring model_path = converter.from_bytes(modelpath);
	m_session = new Ort::Session(m_onnxEnv, model_path.c_str(), session_options);

#ifdef USE_GPU
	Ort::MemoryInfo _memoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
	session_options.SetIntraOpNumThreads(threads);
#else
	Ort::MemoryInfo _memoryInfo("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
	if (useXnnpack)
	{
		//添加xnnpack加速包
		std::unordered_map<std::string, std::string> _xnnpackOption;
		_xnnpackOption.insert(std::make_pair("intra_op_num_threads", std::to_string(threads - 1)));
		session_options.AppendExecutionProvider("XNNPACK", _xnnpackOption);
		////通过将以下内容添加到会话选项来禁用 ORT 操作内线程池旋转：
		session_options.AddConfigEntry("kOrtSessionOptionsConfigAllowIntraOpSpinning", "0");

		session_options.SetIntraOpNumThreads(1);
	}
	else
	{
		session_options.SetIntraOpNumThreads(threads);
	}
#endif // USE_GPU

	m_ortAllocator = new Ort::Allocator(*m_session, _memoryInfo);

	m_binding = new Ort::IoBinding(*m_session);
	// 获取输入输出节点的数量
	size_t _numInputNodes = m_session->GetInputCount();
	size_t _numOutputNodes = m_session->GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;

	// 获取输入输出节点的名称
	for (size_t i = 0; i < _numInputNodes; i++) {
		TensorInfo _info;
		auto input_name = m_session->GetInputNameAllocated(i, allocator);
		auto _inputTypeInfo = m_session->GetInputTypeInfo(i);
		auto _typeAndShapeInfo = _inputTypeInfo.GetTensorTypeAndShapeInfo();
		_info.mShape = _typeAndShapeInfo.GetShape();
		_info.mType = _typeAndShapeInfo.GetElementType();
		_info.mName = input_name.get();
		MemoryMan* _mm = new MemoryMan(_info, m_ortAllocator, _memoryInfo);
		m_inputMemoryVect.push_back(_mm);
		m_binding->BindInput(_info.mName.c_str(), *_mm->mBound);

		std::cout << "inputName:" << _info.mName;
		std::cout << "inputshape:";
		for (auto var : _info.mShape)
		{
			std::cout << (int)var << " ";
		}
	}

	for (size_t i = 0; i < _numOutputNodes; i++) {

		TensorInfo _info;
		auto output_name = m_session->GetOutputNameAllocated(i, allocator);
		auto _outputTypeInfo = m_session->GetOutputTypeInfo(i);
		auto _typeAndShapeInfo = _outputTypeInfo.GetTensorTypeAndShapeInfo();
		_info.mShape = _typeAndShapeInfo.GetShape();
		_info.mType = _typeAndShapeInfo.GetElementType();
		_info.mName = output_name.get();

		MemoryMan* _mm = new MemoryMan(_info, m_ortAllocator, _memoryInfo);
		m_outputMemoryVect.push_back(_mm);
		m_binding->BindOutput(_info.mName.c_str(), *_mm->mBound);

		std::cout << "outputName:" << _info.mName;
		std::cout << "outputshape:";
		for (auto var : _info.mShape)
		{
			std::cout << (int)var << " ";
		}
	}

	return true;
}

bool OnnxModel::processOrtStatus(OrtStatusPtr status_expr)
{
	OrtStatus* _status = (status_expr);
	if (_status != nullptr) {
		const auto& _api = Ort::GetApi();
		std::cout << _api.GetErrorMessage(_status);
		return true;
	}
	return false;
}
#ifdef USE_GPU
bool OnnxModel::initCudaOpentions(const Ort::SessionOptions& option)
{
	const auto& _api = Ort::GetApi();
	OrtCUDAProviderOptionsV2* _cudaOptions = nullptr;
	OrtStatusPtr _statusExpr = _api.CreateCUDAProviderOptions(&_cudaOptions);
	if (processOrtStatus(_statusExpr))
	{
		return 0;
	}
	std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(_api.ReleaseCUDAProviderOptions)>
		_relCudaOptions(_cudaOptions, _api.ReleaseCUDAProviderOptions);

	std::vector<const char*> keys{
	"device_id",
	"has_user_compute_stream",
	"gpu_mem_limit",
	"arena_extend_strategy",
	"cudnn_conv_algo_search",
	"do_copy_in_default_stream",
	"cudnn_conv_use_max_workspace",
	"cudnn_conv1d_pad_to_nc1d" };

	std::vector<const char*> values{
		"0",
		"0",
		"1073741824",
		"kSameAsRequested",
		"DEFAULT",
		"1",
		"1" };

	_statusExpr = _api.UpdateCUDAProviderOptions(_relCudaOptions.get(), keys.data(), values.data(), 6);
	if (processOrtStatus(_statusExpr))
	{
		return 0;
	}

	OrtAllocator* _cudaAllocator = nullptr;
	_statusExpr = _api.GetAllocatorWithDefaultOptions(&_cudaAllocator);
	if (processOrtStatus(_statusExpr))
	{
		return 0;
	}

	char* _cudaOptionsStr = nullptr;
	_statusExpr = _api.GetCUDAProviderOptionsAsString(_relCudaOptions.get(), _cudaAllocator, &_cudaOptionsStr);
	std::string _cudaOpentionsResult;
	if (_cudaOptionsStr != nullptr) {
		_cudaOpentionsResult = std::string(_cudaOptionsStr, strnlen(_cudaOptionsStr, 2048));
		SIM_LOG_INFO << _cudaOpentionsResult;
	}
	_api.AllocatorFree(_cudaAllocator, (void*)_cudaOptionsStr);

	_statusExpr = _api.SessionOptionsAppendExecutionProvider_CUDA_V2
	(static_cast<OrtSessionOptions*>(option), _relCudaOptions.get());
	if (processOrtStatus(_statusExpr))
	{
		return 0;
	}
	//_api.ReleaseCUDAProviderOptions(_cudaOptions);
	return true;
}
#endif