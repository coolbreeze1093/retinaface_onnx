#include <algorithm>
//#include "omp.h"
#include "FaceDetector.h"

Detector::Detector():
        _nms(0.4),
        _threshold(0.6),
        _mean_val{104.f, 117.f, 123.f},
        _retinaface(false)
{

}

inline void Detector::Release(){
    
}

Detector::Detector(const std::string &model_param, const std::string &model_bin, bool retinaface):
        _nms(0.4),
        _threshold(0.6),
        _mean_val{104.f, 117.f, 123.f},
        _retinaface(retinaface)
{
    Init(model_param, model_bin);
}

void Detector::Init(const std::string &model_param, const std::string &model_bin)
{
    OnnxModel::init(model_param, 4, true);
}

void Detector::Detect(cv::Mat& bgr, std::vector<bbox>& boxes)
{
	if (!m_session && m_inputMemoryVect.empty())
	{
		return;
	}
    MemoryMan* _mm = m_inputMemoryVect.at(0);

	float _dstW = 640.0, _dstH = 640.0;
    float _orgW = bgr.cols,_orgH = bgr.rows;
    
    cv::Mat _mat;
    cv::resize(bgr, _mat, cv::Size(_dstW, _dstH));
    cv::Mat _float_mat;
    _mat.convertTo(_float_mat, CV_32FC3);
    cv::Mat mean = cv::Mat(cv::Size(_dstW, _dstH), CV_32FC3, cv::Scalar(104.f, 117.f, 123.f));
    cv::Mat normalized_image = _float_mat - mean;

	cv::Mat _inputIm32F = cv::dnn::blobFromImage(normalized_image);

#ifdef USE_GPU
	cudaMemcpy(_mm->mMemory, m_inputIm32F.data, _mm->mSizeWithByte,
		cudaMemcpyHostToDevice);
#else
	std::memcpy(_mm->mMemory, _inputIm32F.data, _mm->mSizeWithByte);
#endif // USE_GPU
	// 运行模型
	m_session->Run(Ort::RunOptions{}, *m_binding);

    postProcess(m_outputMemoryVect,boxes,_orgW,_orgH,_dstW,_dstH);
}

inline void Detector::SetDefaultParams(){
    _nms = 0.4;
    _threshold = 0.6;
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
}

void Detector::postProcess(const std::vector<MemoryMan*>& resultVec, std::vector<bbox>& boxes
    ,int orgW,int orgH,int dstW,int dstH)
{
	create_anchor_retinaface(m_anchor, dstW, dstH);

	std::vector<bbox> total_box;

	float* ptr = (float*)m_outputMemoryVect.at(0)->mMemory;
	float* ptr1 = (float*)m_outputMemoryVect.at(1)->mMemory;
	float* landms = (float*)m_outputMemoryVect.at(2)->mMemory;

	// #pragma omp parallel for num_threads(2)
	std::vector<cv::Rect2d>_nmsBox;
	std::vector<float>_nmsScore;
	for (int i = 0; i < m_anchor.size(); ++i)
	{
		if (*(ptr1 + 1) > _threshold)
		{
			box tmp = m_anchor[i];
			box tmp1;
			bbox result;

			// loc and conf
			tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
			tmp1.cy = tmp.cy + *(ptr + 1) * 0.1 * tmp.sy;
			tmp1.sx = tmp.sx * exp(*(ptr + 2) * 0.2);
			tmp1.sy = tmp.sy * exp(*(ptr + 3) * 0.2);

			result.x1 = (tmp1.cx - tmp1.sx / 2);
			result.y1 = (tmp1.cy - tmp1.sy / 2);
			result.x2 = (tmp1.cx + tmp1.sx / 2);
			result.y2 = (tmp1.cy + tmp1.sy / 2);
			
			result.s = *(ptr1 + 1);

			// landmark
			for (long long j = 0; j < 5; ++j)
			{
				result.point[j]._x = (tmp.cx + *(landms + (j << 1)) * 0.1 * tmp.sx);
				result.point[j]._y = (tmp.cy + *(landms + (j << 1) + 1) * 0.1 * tmp.sy);
			}
			_nmsBox.push_back(std::move(cv::Rect2d(result.x1, result.y1, result.x2, result.y2)));
			_nmsScore.push_back(result.s);
			total_box.push_back(std::move(result));
		}
		ptr += 4;
		ptr1 += 2;
		landms += 10;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(_nmsBox, _nmsScore, 0.6, _nms, nms_result);

	float _ratioW = float(dstW) / orgW;
	float _ratioH = float(dstH) / orgH;

	for (int j = 0; j < nms_result.size(); ++j)
	{
		bbox _obbox = total_box[nms_result[j]];
		_obbox.x1 *= orgW;
		_obbox.x2 *= orgW;
		_obbox.y1 *= orgH;
		_obbox.y2 *= orgH;
		for (int i = 0; i < 5; ++i)
		{
			_obbox.point[i]._x *= orgW;
			_obbox.point[i]._y *= orgH;
		}
		if (_obbox.x1 < 0)_obbox.x1 = 0;
		if (_obbox.y1 < 0)_obbox.y1 = 0;
		if (_obbox.x2 > orgW)_obbox.x2 = orgW;
		if (_obbox.y2 > orgH)_obbox.y2 = orgH;

		boxes.push_back(std::move(_obbox));
	}
}

Detector::~Detector(){
    Release();
}

void Detector::create_anchor_retinaface(std::vector<box> &anchor, int w, int h)
{
    if (m_lastDstWidth == w || m_lastDstHeight == h)
    {
        return;
    }
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {16, 32};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {64, 128};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {256, 512};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}
