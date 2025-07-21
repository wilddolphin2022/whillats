// client.cc
// Compile with: g++ -std=c++11 -x objective-c++ client.cc -o client -framework Foundation
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/wait.h>
#include <Foundation/Foundation.h>

// Mutex for synchronizing pipe writes
std::mutex pipe_mutex;

// Write WAV file (16 kHz, 1 channel, 16-bit PCM)
void write_wav_file(const std::string& filename, const std::vector<int16_t>& samples, int sample_rate) {
    std::cerr << "Writing WAV file: " << filename << " with " << samples.size() << " samples at " << sample_rate << " Hz\n";
    
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "Failed to open WAV file: " << filename << ": " << strerror(errno) << "\n";
        return;
    }
    
    uint32_t sample_rate_32 = sample_rate;
    uint32_t byte_rate = sample_rate * 1 * 2;
    uint32_t data_size = samples.size() * 2;
    uint32_t chunk_size = data_size + 36;
    
    fwrite("RIFF", 1, 4, fp);
    fwrite(&chunk_size, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);
    fwrite("fmt ", 1, 4, fp);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, fp);
    uint16_t audio_format = 1;
    fwrite(&audio_format, 2, 1, fp);
    uint16_t num_channels = 1;
    fwrite(&num_channels, 2, 1, fp);
    fwrite(&sample_rate_32, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    uint16_t block_align = 1 * 2;
    fwrite(&block_align, 2, 1, fp);
    uint16_t bits_per_sample = 16;
    fwrite(&bits_per_sample, 2, 1, fp);
    fwrite("data", 1, 4, fp);
    fwrite(&data_size, 4, 1, fp);
    
    fwrite(samples.data(), 2, samples.size(), fp);
    fclose(fp);
    
    std::cerr << "WAV file written: " << filename << "\n";
}

void reader_thread(int input_fd) {
    int buffer_count = 0;
    while (buffer_count < 2) {
        uint32_t buffer_size;
        ssize_t bytes_read = read(input_fd, &buffer_size, sizeof(buffer_size));
        if (bytes_read != sizeof(buffer_size)) {
            if (bytes_read == 0) {
                std::cerr << "Reader thread received EOF\n";
                break;
            }
            std::cerr << "Reader thread failed to read buffer size, bytes read: " << bytes_read << ": " << strerror(errno) << "\n";
            break;
        }
        std::cerr << "Reader thread received buffer size: " << buffer_size << "\n";
        if (buffer_size == 0) {
            std::cerr << "Reader thread received end-of-utterance sentinel\n";
            continue;
        }
        
        size_t num_samples = buffer_size / sizeof(int16_t);
        std::vector<int16_t> buffer(num_samples);
        bytes_read = read(input_fd, buffer.data(), buffer_size);
        if (bytes_read != buffer_size) {
            std::cerr << "Reader thread failed to read buffer, bytes read: " << bytes_read << ": " << strerror(errno) << "\n";
            break;
        }
        std::cerr << "Reader thread received buffer: " << num_samples << " samples\n";
        
        int thread_id = buffer_count + 1;
        std::string filename = "synthesized_audio_thread_" + std::to_string(thread_id) + ".wav";
        std::cerr << "Writing buffer to " << filename << " with " << num_samples << " samples\n";
        write_wav_file(filename, buffer, 16000);
        
        buffer_count++;
    }
    std::cerr << "Reader thread exiting\n";
}

void sender_thread(int output_fd, const std::string& text, const std::string& language, int thread_id) {
    std::cerr << "Thread " << thread_id << " sending text: " << text << "\n";
    std::lock_guard<std::mutex> lock(pipe_mutex);
    uint32_t text_size = text.size();
    if (write(output_fd, &text_size, sizeof(text_size)) != sizeof(text_size)) {
        std::cerr << "Thread " << thread_id << " failed to write text size: " << strerror(errno) << "\n";
        return;
    }
    if (write(output_fd, text.data(), text_size) != text_size) {
        std::cerr << "Thread " << thread_id << " failed to write text: " << strerror(errno) << "\n";
        return;
    }
    uint32_t lang_size = language.size();
    if (write(output_fd, &lang_size, sizeof(lang_size)) != sizeof(lang_size)) {
        std::cerr << "Thread " << thread_id << " failed to write language size: " << strerror(errno) << "\n";
        return;
    }
    if (write(output_fd, language.data(), lang_size) != lang_size) {
        std::cerr << "Thread " << thread_id << " failed to write language: " << strerror(errno) << "\n";
        return;
    }
    std::cerr << "Thread " << thread_id << " sent text\n";
}

int main() {
    std::cerr << "Client process started, PID: " << getpid() << "\n";
    
    int pipe_to_child[2], pipe_from_child[2];
    if (pipe(pipe_to_child) == -1 || pipe(pipe_from_child) == -1) {
        std::cerr << "Failed to create pipes: " << strerror(errno) << "\n";
        return 1;
    }
    std::cerr << "Pipes created\n";
    
    pid_t pid = fork();
    if (pid == -1) {
        std::cerr << "Failed to fork: " << strerror(errno) << "\n";
        return 1;
    }
    
    if (pid > 0) {
        close(pipe_to_child[0]);
        close(pipe_from_child[1]);
        std::cerr << "Parent process: output_fd=" << pipe_to_child[1] << ", input_fd=" << pipe_from_child[0] << "\n";
        
        std::thread reader(reader_thread, pipe_from_child[0]);
        
        // Send texts sequentially along with language codes to enforce order
        sender_thread(pipe_to_child[1], "Hello from thread one.", "en-US", 1);
        sender_thread(pipe_to_child[1], "Greetings from thread two.", "ru-RU", 2);
        
        close(pipe_to_child[1]);
        
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            std::cerr << "Failed to wait for child: " << strerror(errno) << "\n";
        } else if (WIFEXITED(status)) {
            std::cerr << "Child process exited with status: " << WEXITSTATUS(status) << "\n";
        } else {
            std::cerr << "Child process terminated abnormally\n";
        }
        
        reader.join();
        close(pipe_from_child[0]);
        
        std::cerr << "Client process exiting, PID: " << getpid() << "\n";
        return 0;
    } else {
        close(pipe_to_child[1]);
        close(pipe_from_child[0]);
        std::cerr << "Child process started, PID: " << getpid() << "\n";
        
        if (dup2(pipe_to_child[0], STDIN_FILENO) == -1) {
            std::cerr << "Failed to redirect stdin: " << strerror(errno) << "\n";
            exit(1);
        }
        if (dup2(pipe_from_child[1], STDOUT_FILENO) == -1) {
            std::cerr << "Failed to redirect stdout: " << strerror(errno) << "\n";
            exit(1);
        }
        close(pipe_to_child[0]);
        close(pipe_from_child[1]);
        
        std::cerr << "Child process launching synthesis\n";
        if (access("./synthesis", X_OK) != 0) {
            std::cerr << "Cannot execute synthesis: " << strerror(errno) << "\n";
            exit(1);
        }
        execl("./synthesis", "./synthesis", nullptr);
        std::cerr << "Failed to launch synthesis: " << strerror(errno) << "\n";
        exit(1);
    }
}