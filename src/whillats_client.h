#ifndef WHILLATS_CLIENT_H
#define WHILLATS_CLIENT_H

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <sys/types.h>

#include "whillats.h"
#include "whillats_ipc.h"

class WhillatsServerConnection {
public:
    WhillatsServerConnection();
    ~WhillatsServerConnection();

    bool start(const std::string& server_path, const whillats_ipc::ConfigMsg& cfg);
    void stop();
    bool isRunning() const { return _running; }

    bool sendMsg(uint8_t type, const void* data, uint32_t len);

    void setWhisperCallback(WhillatsSetResponseCallback cb) { _whisperCb = cb; }
    void setLanguageCallback(WhillatsSetLanguageCallback cb) { _langCb = cb; }
    void setLlamaCallback(WhillatsSetResponseCallback cb) { _llamaCb = cb; }
    void setTtsCallback(WhillatsSetAudioCallback cb) { _ttsCb = cb; }

private:
    void readerThread();

    pid_t _serverPid = -1;
    int _writeFd = -1;
    int _readFd = -1;
    std::atomic<bool> _running{false};
    std::thread _reader;
    std::mutex _writeMutex;

    WhillatsSetResponseCallback _whisperCb{nullptr, nullptr};
    WhillatsSetLanguageCallback _langCb{nullptr, nullptr};
    WhillatsSetResponseCallback _llamaCb{nullptr, nullptr};
    WhillatsSetAudioCallback    _ttsCb{nullptr, nullptr};
};

class WhillatsTranscriberClient {
public:
    WhillatsTranscriberClient(WhillatsServerConnection& conn);
    ~WhillatsTranscriberClient();

    bool start();
    void stop();
    void processAudioBuffer(uint8_t* buffer, size_t size);
    std::string getLanguage() const { return _language; }
    void setLanguage(const char* lang) { if (lang) _language = lang; }
    void setThreadCount(int) {}

private:
    WhillatsServerConnection& _conn;
    std::string _language = "en";
    bool _started = false;
};

class WhillatsLlamaClient {
public:
    WhillatsLlamaClient(WhillatsServerConnection& conn);
    ~WhillatsLlamaClient();

    bool start();
    void stop();
    bool isRunning() const { return _started; }
    void setThreadCount(int) {}
    void askLlama(const char* prompt);
    void receiveVideoFrame(const YUVData& yuv);

private:
    WhillatsServerConnection& _conn;
    bool _started = false;
};

class WhillatsTTSClient {
public:
    WhillatsTTSClient(WhillatsServerConnection& conn);
    ~WhillatsTTSClient();

    bool start();
    void stop();
    void queueText(const char* text, const char* language);
    static int getSampleRate() { return 16000; }

private:
    WhillatsServerConnection& _conn;
    bool _started = false;
};

#endif  // WHILLATS_CLIENT_H
