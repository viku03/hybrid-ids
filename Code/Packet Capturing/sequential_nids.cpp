#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <string>
#include <cmath>
#include <memory>
#include <algorithm>
#include <torch/torch.h>
#include <torch/script.h>
#include <filesystem>
#include <random>
#include <fstream>

// [Previous Flow struct remains the same]
struct Flow {
    std::string srcIP;
    std::string dstIP;
    uint16_t srcPort;
    uint16_t dstPort;
    uint8_t protocol;
    std::vector<double> timestamps;
    std::vector<size_t> byteCounts;
    std::vector<float> features;
    double meanIAT;
    double stdIAT;
    double byteRatio;
};

class SequentialNetworkTransformer {
private:
    torch::jit::script::Module module;
    torch::Device device{torch::kCPU};
    const int input_dim = 78;
    
public:
    SequentialNetworkTransformer() {
        try {
            // Device selection (simplified)
            if (torch::cuda::is_available()) {
                device = torch::kCUDA;
            } else {
                device = torch::kCPU;
            }

            // Load model (same as previous implementation)
            if (!std::filesystem::exists("/Users/Viku/GitHub/hybrid-ids/Code/model/best_model.pt")) {
                throw std::runtime_error("Model file not found");
            }

            module = torch::jit::load("/Users/Viku/GitHub/hybrid-ids/Code/model/best_model.pt");
            module.to(device);
            module.eval();
            
        } catch (const std::exception& e) {
            std::cerr << "Error initializing model: " << e.what() << std::endl;
            throw;
        }
    }
    
    torch::Tensor forward(torch::Tensor x) {
        try {
            if (x.sizes().back() != input_dim) {
                throw std::runtime_error("Input tensor dimension mismatch");
            }
            
            x = x.to(device).to(torch::kFloat32);
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(x);
            
            torch::NoGradGuard no_grad;
            auto output = module.forward(inputs).toTensor();
            
            return torch::log_softmax(output, 1);
            
        } catch (const std::exception& e) {
            std::cerr << "Error in forward pass: " << e.what() << std::endl;
            throw;
        }
    }

    std::vector<float> extractFeatures(const Flow& flow) {
        std::vector<float> features(78, 0.0f);
        
        std::vector<double> iats;
        if (flow.timestamps.size() > 1) {
            for (size_t i = 1; i < flow.timestamps.size(); ++i) {
                iats.push_back(flow.timestamps[i] - flow.timestamps[i-1]);
            }
        }
        
        // [Rest of feature extraction remains the same as parallel version]
        features[0] = static_cast<float>(flow.protocol);
        features[1] = static_cast<float>(flow.timestamps.size());
        
        size_t total_bytes = 0;
        for (size_t bytes : flow.byteCounts) {
            total_bytes += bytes;
        }
        features[2] = static_cast<float>(total_bytes);
        features[3] = flow.byteCounts.empty() ? 0.0f : 
                    static_cast<float>(total_bytes) / flow.byteCounts.size();
        
        if (!iats.empty()) {
            double mean_iat = std::accumulate(iats.begin(), iats.end(), 0.0) / iats.size();
            features[4] = static_cast<float>(mean_iat);
            
            double sq_sum = 0.0;
            for (double iat : iats) {
                sq_sum += (iat - mean_iat) * (iat - mean_iat);
            }
            features[5] = static_cast<float>(std::sqrt(sq_sum / iats.size()));
        }
        
        // Packet lengths and IATs
        for (size_t i = 0; i < std::min(flow.byteCounts.size(), size_t(36)); ++i) {
            features[i + 6] = static_cast<float>(flow.byteCounts[i]);
        }
        
        for (size_t i = 0; i < std::min(iats.size(), size_t(36)); ++i) {
            features[i + 42] = static_cast<float>(iats[i]);
        }
        
        return features;
    }
};

class SequentialNIDS {
private:
    std::unordered_map<std::string, Flow> flowTable;
    std::unique_ptr<SequentialNetworkTransformer> model;
    
    std::string calculateFlowHash(const Flow& flow) {
        return flow.srcIP + ":" + 
               std::to_string(flow.srcPort) + "-" + 
               flow.dstIP + ":" + 
               std::to_string(flow.dstPort) + "-" + 
               std::to_string(flow.protocol);
    }
    
    bool isAnomaly(const Flow& flow) {
        try {
            auto features = model->extractFeatures(flow);
            
            auto input = torch::from_blob(features.data(), {1, 78}, torch::kFloat);
            
            torch::NoGradGuard no_grad;
            auto output = model->forward(input);
            
            auto prediction = output.argmax(1);
            return prediction.item<int>() == 1;
            
        } catch (const std::exception& e) {
            std::cerr << "Error in anomaly detection: " << e.what() << std::endl;
            return false;
        }
    }

    std::unordered_map<std::string, std::vector<uint16_t>> source_port_scans;
    const size_t PORT_SCAN_THRESHOLD = 5;

    void checkPortScan(const std::string& srcIP, uint16_t dstPort) {
        source_port_scans[srcIP].push_back(dstPort);
        
        if (source_port_scans[srcIP].size() >= PORT_SCAN_THRESHOLD) {
            std::cout << "\nPOTENTIAL PORT SCAN DETECTED:\n"
                      << "Source IP: " << srcIP << "\n"
                      << "Unique ports scanned: " << source_port_scans[srcIP].size() << "\n";
        }
    }

    std::vector<double> processingTimes;
    std::vector<double> anomalyDetectionTimes;

public:
    SequentialNIDS() : model(std::make_unique<SequentialNetworkTransformer>()) {}
    
    double calculateMeanIAT(const Flow& flow) {
        if (flow.timestamps.size() < 2) return 0.0;
        
        double sum = 0.0;
        for (size_t i = 1; i < flow.timestamps.size(); ++i) {
            sum += flow.timestamps[i] - flow.timestamps[i-1];
        }
        return sum / (flow.timestamps.size() - 1);
    }
    
    double calculateStdIAT(const Flow& flow) {
        if (flow.timestamps.size() < 2) return 0.0;
        
        double mean = calculateMeanIAT(flow);
        double sum_sq = 0.0;
        
        for (size_t i = 1; i < flow.timestamps.size(); ++i) {
            double iat = flow.timestamps[i] - flow.timestamps[i-1];
            sum_sq += (iat - mean) * (iat - mean);
        }
        
        return std::sqrt(sum_sq / (flow.timestamps.size() - 1));
    }
    
    double calculateByteRatio(const Flow& flow) {
        if (flow.byteCounts.empty()) return 0.0;
        
        size_t total = 0;
        for (size_t bytes : flow.byteCounts) {
            total += bytes;
        }
        return static_cast<double>(total) / flow.byteCounts.size();
    }
    
    void reportAnomaly(const Flow& flow) {
        std::cout << "Anomaly detected in flow: " 
                  << flow.srcIP << ":" << flow.srcPort << " -> "
                  << flow.dstIP << ":" << flow.dstPort << "\n"
                  << "Protocol: " << static_cast<int>(flow.protocol) << "\n"
                  << "Mean IAT: " << flow.meanIAT << " ms\n"
                  << "Std IAT: " << flow.stdIAT << " ms\n"
                  << "Byte Ratio: " << flow.byteRatio << "\n"
                  << "Packet Count: " << flow.timestamps.size() << "\n\n";
    }
     void logBenchmarkResults(const std::string& filename) {
        std::ofstream outFile(filename);
        
        if (!outFile.is_open()) {
            std::cerr << "Failed to open benchmark log file: " << filename << std::endl;
            return;
        }

        // Basic statistics calculation
        double total_processing_time = 0.0;
        double total_anomaly_detection_time = 0.0;
        
        for (double time : processingTimes) {
            total_processing_time += time;
        }
        
        for (double time : anomalyDetectionTimes) {
            total_anomaly_detection_time += time;
        }

        double avg_processing_time = total_processing_time / processingTimes.size();
        double avg_anomaly_detection_time = total_anomaly_detection_time / anomalyDetectionTimes.size();

        outFile << "Benchmark Results (Sequential):\n"
                << "Total Packets Processed: " << processingTimes.size() << "\n"
                << "Average Packet Processing Time: " << avg_processing_time << " ms\n"
                << "Average Anomaly Detection Time: " << avg_anomaly_detection_time << " ms\n"
                << "\nDetailed Processing Times:\n";

        for (size_t i = 0; i < processingTimes.size(); ++i) {
            outFile << "Packet " << i+1 << ": "
                    << "Processing Time = " << processingTimes[i] << " ms, "
                    << "Anomaly Detection Time = " << anomalyDetectionTimes[i] << " ms\n";
        }

        // Additional system information
        outFile << "\nSystem Information:\n"
                << "Processing Approach: Sequential\n"
                << "Threads Used: 1\n";

        outFile.close();
    }

    void processPacket(const std::string& srcIP, const std::string& dstIP,
                  uint16_t srcPort, uint16_t dstPort, uint8_t protocol,
                  size_t byteCount) {
        auto packet_start_time = std::chrono::high_resolution_clock::now();

        try {
            // Port scan check
            checkPortScan(srcIP, dstPort);

            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
            
            std::string flowHash = srcIP + ":" + 
                                std::to_string(srcPort) + "-" + 
                                dstIP + ":" + 
                                std::to_string(dstPort) + "-" + 
                                std::to_string(protocol);
            
            auto& flow = flowTable[flowHash];
            
            if (flow.timestamps.empty()) {
                flow.srcIP = srcIP;
                flow.dstIP = dstIP;
                flow.srcPort = srcPort;
                flow.dstPort = dstPort;
                flow.protocol = protocol;
            }
            
            flow.timestamps.push_back(timestamp);
            flow.byteCounts.push_back(byteCount);
            
            if (flow.timestamps.size() > 1) {
                // Sequential computation of flow statistics
                flow.meanIAT = calculateMeanIAT(flow);
                flow.stdIAT = calculateStdIAT(flow);
                flow.byteRatio = calculateByteRatio(flow);
                
                // Benchmark anomaly detection
                auto anomaly_start_time = std::chrono::high_resolution_clock::now();
                bool is_anomaly = isAnomaly(flow);
                auto anomaly_end_time = std::chrono::high_resolution_clock::now();
                
                double anomaly_detection_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    anomaly_end_time - anomaly_start_time).count() / 1000.0;
                anomalyDetectionTimes.push_back(anomaly_detection_time);
                
                if (is_anomaly) {
                    reportAnomaly(flow);
                }
            }
            
            // Calculate and store processing time
            auto packet_end_time = std::chrono::high_resolution_clock::now();
            double processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
                packet_end_time - packet_start_time).count() / 1000.0;
            processingTimes.push_back(processing_time);
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing packet: " << e.what() << std::endl;
        }
    }
};

int main() {
    try {
        SequentialNIDS nids;
        
        auto total_start_time = std::chrono::high_resolution_clock::now();
        
        // Simplified test case - generate more packets for better benchmarking
        std::string src_ip = "192.168.1.100";
        std::string dst_ip = "10.0.0.1";
        uint16_t src_port = 80;
        uint16_t dst_port = 443;
        uint8_t protocol = 6;  // TCP
        
        // Generate 20 packets with varying characteristics
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> byte_dist(50, 1500);
        
        for (int i = 0; i < 500; ++i) {
            size_t byte_count = byte_dist(gen);
            
            nids.processPacket(src_ip, dst_ip, src_port, dst_port, protocol, byte_count);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Reduced sleep for more packets
        }
        
        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            total_end_time - total_start_time);
        
        std::cout << "Total Execution Time: " << total_duration.count() << " milliseconds" << std::endl;
        
        // Log benchmark results
        nids.logBenchmarkResults("sequential_nids_benchmark_results.txt");
        
    } catch (const std::exception& e) {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}