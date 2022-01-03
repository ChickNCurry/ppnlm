#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace cv;

class ppnlm_denoiser {

    const bool store_deltas = true;

    const float smoothing = 1;
    const int patch_size = 3;
    const int kernel_size = 3;
    int rel_max = (int) pow(2 * kernel_size + 1, 2);
    const int max_depth = 8;
    const float rel_thresh = 12;

    vector<Mat> downfuse_levels;
    vector<Mat> upfuse_levels;
    vector<Mat> rel_levels;
    vector<vector<Mat>> delta_levels;

    // compute the largest possible size of a patch/kernel of mat centered around x,y and store the results in the refs
    void computeMinMax(Mat &mat, int y, int x, int &min_y, int &min_x, int &max_y, int &max_x, int size) {

        min_y = max(y - size, 0);
        min_x = max(x - size, 0);

        max_y = min(y + size, mat.rows);
        max_x = min(x + size, mat.cols);
    }

    // compute the largest possible patch in mat centered around x,y
    Mat getPatch(Mat &mat, int y, int x) {

        int min_y, min_x, max_y, max_x;
        computeMinMax(mat, y, x, min_y, min_x, max_y, max_x, patch_size);

        Range rows(min_y, max_y);
        Range cols(min_x, max_x);

        Mat out(mat, rows, cols);
        return out;
    }

    // compute a delta by comparing 2 patches
    float calculateDelta(Mat &local_patch, Mat &other_patch) {

        float sum = 0;

        for(int y = 0; y < local_patch.rows; y++) {
            for(int x = 0; x < local_patch.cols; x++) {

                float diff = (1 / smoothing) * (local_patch.at<float>(y, x) - other_patch.at<float>(y, x));
                sum += (float) pow(diff, 2);
            }
        }

        return exp(-sum);
    }

    //compute the current downfuse pixel
    float downfusePixel(Mat mat, int local_y, int local_x, float &weight_sum, float &weight_sqrd_sum, Mat &delta_mat) {

        float pixel_sum = 0;

        Mat local_patch = getPatch(mat, local_y, local_x);

        int min_y, min_x, max_y, max_x;
        computeMinMax(mat, local_y, local_x, min_y, min_x, max_y, max_x, kernel_size);

        delta_mat = Mat(max_y - min_y, max_x - min_x, mat.type());

        for(int y = min_y; y < max_y; y++) {
            for(int x = min_x; x < max_x; x++) {

                Mat other_patch = getPatch(mat, y, x);

                float rel_score = rel_levels.back().at<float>(y, x);

                float delta = calculateDelta(local_patch, other_patch);
                assert((delta >= 0 && delta <= 1));

                float weight = rel_score * delta;
                assert((weight >= 0 && weight <= rel_max));

                delta_mat.at<float>(y - min_y, x - min_x) = delta;

                weight_sum += weight;
                weight_sqrd_sum += (float) pow(weight, 2);

                pixel_sum += weight * mat.at<float>(y, x);
            }
        }

        return pixel_sum;
    }

    // compute the current downfuse level
    void downfuse(int depth) {

        cout << "downfuse level: " << depth << endl;
        if(depth >= max_depth) return;

        Mat input = downfuse_levels.back();
        Mat output(input.rows / 2, input.cols / 2, input.type());
        Mat rels(output.rows, output.cols, output.type());
        vector<Mat> deltas(input.rows * input.cols);

        #pragma omp parallel for
        for(int y = 0; y < input.rows; y += 2) {
            for(int x = 0; x < input.cols; x += 2) {

                float weight_sum = 0;
                float weight_sqrd_sum = 0;
                Mat delta_mat;

                float val = downfusePixel(input, y, x, weight_sum, weight_sqrd_sum, delta_mat);

                deltas[(y * input.rows + x) / 2] = delta_mat;

                if(weight_sum != 0) {
                    float val_output = val / weight_sum;
                    assert((val_output >= 0 && val_output <= 1));
                    output.at<float>(y / 2, x / 2) = val_output;
                } else {
                    output.at<float>(y / 2, x / 2) = 0;
                }

                if(weight_sqrd_sum != 0) {
                    float rel_output = ((float) pow(weight_sum, 2)) / weight_sqrd_sum;
                    assert((rel_output >= 1 && rel_output <= rel_max));
                    rels.at<float>(y / 2, x / 2) = rel_output;
                } else {
                    rels.at<float>(y / 2, x / 2) = 1;
                }
            }
        }

        downfuse_levels.push_back(output);
        //normalizeRels(rels);
        rel_levels.push_back(rels);
        delta_levels.push_back(deltas);

        downfuse(++depth);
    }

    // compute the nlm part of a pixel in the current upfuse level
    float nlmPixel(Mat &mat, int local_y, int local_x, float &weight_sum, Mat &deltas) {

        float pixel_sum = 0;

        Mat local_patch = getPatch(mat, local_y, local_x);

        int min_y, min_x, max_y, max_x;
        computeMinMax(mat, local_y, local_x, min_y, min_x, max_y, max_x, kernel_size);

        for(int y = min_y; y < max_y; y++) {
            for(int x = min_x; x < max_x; x++) {

                float rel_score = rel_levels[rel_levels.size() - 2].at<float>(y, x);

                float delta;
                if(store_deltas && local_y % 2 == 0 && local_x % 2 == 0) {
                    delta = deltas.at<float>(y - min_y, x - min_x);
                } else {
                    Mat other_patch = getPatch(mat, y, x);
                    delta = calculateDelta(local_patch, other_patch);
                    assert((delta >= 0 && delta <= 1));
                }

                float weight = rel_score * delta;
                //float weight = delta;
                assert((weight >= 0 && weight <= rel_max));

                weight_sum += weight;
                pixel_sum += weight * mat.at<float>(y, x);
            }
        }

        return pixel_sum;
    }

    // compute the upfuse part of a pixel in the current upfuse level
    float upfusePixel(Mat &finer, Mat &coarser, int local_y, int local_x, float &weight_sum, Mat &deltas) {

        float pixel_sum = 0;

        Mat local_patch = getPatch(finer, local_y, local_x);

        int min_y, min_x, max_y, max_x;
        computeMinMax(finer, local_y, local_x, min_y, min_x, max_y, max_x, kernel_size);

        for(int y = min_y; y < max_y; y++) {
            for(int x = min_x; x < max_x; x++) {

                float rel_score = rel_levels.back().at<float>(y / 2, x / 2);

                float delta;
                if(store_deltas && local_y % 2 == 0 && local_x % 2 == 0) {
                    delta = deltas.at<float>(y - min_y, x - min_x);
                } else {
                    Mat other_patch = getPatch(finer, y, x);
                    delta = calculateDelta(local_patch, other_patch);
                    assert((delta >= 0 && delta <= 1));
                }

                float weight = rel_score * delta;
                assert((weight >= 0 && weight <= rel_max));

                weight_sum += weight;
                pixel_sum += weight * coarser.at<float>(y / 2, x / 2);
            }
        }

        return pixel_sum;
    }

    // compute the current upfuse level
    void upfuse(int depth) {

        cout << "upfuse level: " << depth << endl;
        if(depth <= 0) return;

        Mat finer = downfuse_levels[depth - 1];
        Mat coarser = upfuse_levels.back();
        Mat output(finer.rows, finer.cols, finer.type());

        vector<Mat> deltas = delta_levels.back();

        #pragma omp parallel for
        for(int y = 0; y < output.rows; y++) {
            for(int x = 0; x < output.cols; x++) {

                float weight_sum = 0;

                Mat delta_mat = deltas[(y * output.rows + x) / 2];

                float val2 = upfusePixel(finer, coarser, y, x, weight_sum, delta_mat);
                float val1 = nlmPixel(finer, y, x, weight_sum, delta_mat);

                if(weight_sum != 0) {
                    float val_output = (val1 + val2) / weight_sum;
                    assert((val_output >= 0 && val_output <= 1));
                    output.at<float>(y, x) = val_output;
                } else {
                    output.at<float>(y, x) = 0;
                }
            }
        }

        upfuse_levels.push_back(output);
        //writeRels(rel_levels.back());
        rel_levels.pop_back();
        delta_levels.pop_back();

        upfuse(--depth);
    }

    void normalizeRels(Mat &rels) {
        rels = rels / rel_max;
    }

    void threshRels(Mat &rels) {
        for(int i = 0; i < rels.rows; i++) {
            for(int j = 0; j < rels.cols; j++) {
                //float val = rels.at<float>(i, j) * rel_max - rel_thresh;
                float val = rels.at<float>(i, j) - rel_thresh;
                rels.at<float>(i, j) = fmax(val, 0);
            }
        }
    }

public:

    // denoise input mat and return the result
    Mat denoise(Mat &mat) {

        downfuse_levels.push_back(mat);
        rel_levels.push_back(Mat(mat.rows, mat.cols, mat.type(), rel_max));

        downfuse(0);

        upfuse_levels.push_back(downfuse_levels.back());

        for(auto rels: rel_levels) {
            threshRels(rels);
            //normalizeRels(rels);
        }

        upfuse(max_depth);

        return upfuse_levels.back();
    }

    // write the downfuse and upfuse level mats to images
    void writeStages() {

        cout << "writing downfuse levels" << endl;
        for(Mat downfuse : downfuse_levels) {
            Mat output;
            downfuse.convertTo(output, CV_8U, 255.0);
            imwrite("images/levels/downfuse_" + to_string(output.rows) + ".png", output);
        }

        cout << "writing upfuse levels" << endl;
        for(Mat upfuse : upfuse_levels) {
            Mat output;
            upfuse.convertTo(output, CV_8U, 255.0);
            imwrite("images/levels/upfuse_" + to_string(output.rows) + ".png", output);
        }
    }

    void writeRels(Mat &rels) {

        cout << "writing rels" << endl;

        Mat output;
        rels.convertTo(output, CV_8U, 255.0);

        imwrite("images/rels/rels_" + to_string(output.rows) + ".png", output);
    }
};


int main() {

    // measure starting moment
    auto start = chrono::high_resolution_clock::now();

    string image_path = "images/source/gnoise.png";
    Mat input, float_input, float_output, output;

    // load input uint image and convert to a float image
    input = imread(image_path, IMREAD_GRAYSCALE);
    input.convertTo(float_input, CV_32F, 1.0 / 255.0);

    // denoise image
    ppnlm_denoiser ppnlm;
    float_output = ppnlm.denoise(float_input);

    // measure ending moment and print the duration
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "duration (ms): " << duration.count() << endl;

    // write stages as images
    //ppnlm.writeStages();

    // write output image
    float_output.convertTo(output, CV_8U, 255.0);
    imwrite("images/result/test.png", output);

    return 0;
}
