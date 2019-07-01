// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2019 Nikita Kovalenko <kov.nikit@gmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tf_utils.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>

cv::Mat createHeatmap( const cv::Mat& heatmaps );

int main() {
  TF_Graph* graph = tf_utils::LoadGraph("graph.pb");
  if (graph == nullptr) {
    std::cout << "Can't load graph" << std::endl;
    return 1;
  }

  // prepare input data:
  cv::Mat image = cv::imread( "image.jpg", -1 );

  // convert image to float32
  cv::Mat image32f;
  image.convertTo( image32f, CV_32F );

  // write it to a vector:
  std::vector<float> input_data;
  input_data.assign( (float*) image32f.data, (float*) image32f.data + image32f.total() * image32f.channels() ); 

  // prepare input tensor
  const std::vector<std::int64_t> input_dims = { 1, image.rows, image.cols, image.channels() };
  const std::vector<TF_Output> input_ops = { {TF_GraphOperationByName( graph, "img_input" ), 0} };
  const std::vector<TF_Tensor*> input_tensors = { tf_utils::CreateTensor( TF_FLOAT, input_dims, input_data ) };

  // prepare output tensor
  const std::vector<std::int64_t> output_dims = { 1, image.rows / 8, image.cols / 8, 19 };
  const std::vector<TF_Output> out_ops = { {TF_GraphOperationByName( graph, "net_output" ), 0} };
  std::vector<TF_Tensor*> output_tensors = { nullptr };

  // create TF session:
  auto session = tf_utils::CreateSession( graph, tf_utils::CreateSessionOptions( 0.3f ) );
  if ( session == nullptr ) {
    std::cout << "Can't create session" << std::endl;
    return 2;
  }

  // run the prediction:  
  const TF_Code code = tf_utils::RunSession( session, input_ops, input_tensors, out_ops, output_tensors );

  if ( code == TF_OK ) {    
    // get tensor data
    const std::vector<std::vector<float>> data = tf_utils::GetTensorsData<float>( output_tensors );

    // `data` now contains a float32 vector of model's outputs.
    // this is enough, if we just need to run a prediction and get some probabilities
    // or classify an image

    // ...

    // but if we expect the model's output to also be an image (e.g. a heatmap for keypoint detection)
    // then we need to convert the model output back into a cv::Mat
    cv::Mat output_image( static_cast<int>(output_dims[1]), static_cast<int>(output_dims[2]), CV_32FC( output_dims[3] ), (void*) data.at( 0 ).data() );  

    // do something with the image:
    cv::Mat heatmap = createHeatmap( output_image );

    while ( cv::waitKey( 1 ) != 27 ) { 
      cv::imshow( "original", image );
      cv::imshow( "heatmaps", heatmap );
    }
    cv::destroyAllWindows();

  }
  else {
    std::cout << "Error run session TF_CODE: " << code;
    return code;
  }

  tf_utils::DeleteGraph(graph);
  tf_utils::DeleteTensors(input_tensors);
  tf_utils::DeleteTensors(output_tensors);
    
  return 0;
}
    
cv::Mat createHeatmap( const cv::Mat& heatmaps ) {
  cv::Mat hue_ch = cv::Mat::zeros( heatmaps.rows, heatmaps.cols, CV_8U );
  cv::Mat sat_ch = cv::Mat::ones( heatmaps.rows, heatmaps.cols, CV_8U ) * 255;
  cv::Mat val_ch = cv::Mat::ones( heatmaps.rows, heatmaps.cols, CV_8U ) * 255;

  for ( int i = 0; i < heatmaps.channels() - 1; i++ ) {
    cv::Mat h_ch, h_ch_uint8;
    cv::extractChannel( heatmaps, h_ch, i );

    h_ch *= 180;

    h_ch.convertTo( h_ch_uint8, CV_8U );

    hue_ch |= h_ch_uint8;
  }

  cv::Mat prettyHeatmap;
  cv::merge( std::vector<cv::Mat> { hue_ch, sat_ch, val_ch }, prettyHeatmap );

  cv::cvtColor( prettyHeatmap, prettyHeatmap, cv::COLOR_HSV2RGB );

  return prettyHeatmap;
}