#pragma once
#include <vector>

namespace mediapipe {
	struct ImageTransforRequir
	{
		enum class ShapeFormat
		{
			nhwc = 0,
			nchw = 1
		};
		enum class ColorMode
		{
			rgb = 0,
			bgr = 1
		};

		ShapeFormat imageShapeFormat;
		std::vector<int64_t> imageShape;
		ColorMode imageColorMode;

	};
}