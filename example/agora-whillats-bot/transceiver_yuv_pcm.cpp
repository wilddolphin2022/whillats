#include <csignal>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <memory>

#include "AgoraBase.h"
#include "AgoraMediaBase.h"
#include "AgoraRefCountedObject.h"
#include "IAgoraService.h"
#include "NGIAgoraRtcConnection.h"
#include "NGIAgoraAudioTrack.h"
#include "NGIAgoraLocalUser.h"
#include "NGIAgoraMediaNodeFactory.h"
#include "NGIAgoraMediaNode.h"
#include "NGIAgoraVideoTrack.h"
#include "NGIAgoraLocalUser.h"
#include "NGIAgoraAudioTrack.h"
#include "NGIAgoraVideoMixerSource.h"

#include "common/helper.h"
#include "common/log.h"
#include "common/opt_parser.h"
#include "common/sample_common.h"
#include "common/sample_connection_observer.h"

#include "whillats.h"

#define DEFAULT_CONNECT_TIMEOUT_MS (3000)
#define DEFAULT_SAMPLE_RATE (16000)
#define DEFAULT_NUM_OF_CHANNELS (1)
#define DEFAULT_TARGET_BITRATE (1 * 1000 * 1000)
#define DEFAULT_VIDEO_WIDTH (352)
#define DEFAULT_VIDEO_HEIGHT (288)
#define DEFAULT_FRAME_RATE (15)
#define STREAM_TYPE_HIGH "high"
#define STREAM_TYPE_LOW "low"

// Forward declaration
class TransceiverYuvPcm;

// --- Embedded SampleLocalUserObserver Definition (BEFORE TransceiverYuvPcm) ---
class SampleLocalUserObserver : public agora::rtc::ILocalUserObserver,
                                public agora::media::IAudioFrameObserverBase,
                                public agora::rtc::IVideoFrameObserver2
{
public:
  SampleLocalUserObserver(TransceiverYuvPcm *transceiver) : transceiver_(transceiver) {}
  virtual ~SampleLocalUserObserver();

  void setLocalUser(agora::rtc::ILocalUser *user);

  // --- ILocalUserObserver overrides ---
  void onUserAudioTrackSubscribed(agora::user_id_t userId, agora::agora_refptr<agora::rtc::IRemoteAudioTrack> audioTrack) override;
  void onUserVideoTrackSubscribed(agora::user_id_t userId, const agora::rtc::VideoTrackInfo &trackInfo, agora::agora_refptr<agora::rtc::IRemoteVideoTrack> videoTrack) override;
  void onAudioTrackPublishSuccess(agora::agora_refptr<agora::rtc::ILocalAudioTrack> audioTrack) override {}
  void onAudioTrackPublicationFailure(agora::agora_refptr<agora::rtc::ILocalAudioTrack> audioTrack, agora::ERROR_CODE_TYPE error) override {}
  void onUserAudioTrackStateChanged(agora::user_id_t userId, agora::agora_refptr<agora::rtc::IRemoteAudioTrack> audioTrack, agora::rtc::REMOTE_AUDIO_STATE state, agora::rtc::REMOTE_AUDIO_STATE_REASON reason, int elapsed) override {}
  void onVideoTrackPublishSuccess(agora::agora_refptr<agora::rtc::ILocalVideoTrack> videoTrack) override {}
  void onVideoTrackPublicationFailure(agora::agora_refptr<agora::rtc::ILocalVideoTrack> videoTrack, agora::ERROR_CODE_TYPE error) override {}
  void onUserVideoTrackStateChanged(agora::user_id_t userId, agora::agora_refptr<agora::rtc::IRemoteVideoTrack> videoTrack, agora::rtc::REMOTE_VIDEO_STATE state, agora::rtc::REMOTE_VIDEO_STATE_REASON reason, int elapsed) override {}
  void onRemoteVideoTrackStatistics(agora::agora_refptr<agora::rtc::IRemoteVideoTrack> videoTrack, const agora::rtc::RemoteVideoTrackStats &stats) override {}
  void onLocalVideoTrackStateChanged(agora::agora_refptr<agora::rtc::ILocalVideoTrack> videoTrack, agora::rtc::LOCAL_VIDEO_STREAM_STATE state, agora::rtc::LOCAL_VIDEO_STREAM_REASON reason) override {}
  void onLocalVideoTrackStatistics(agora::agora_refptr<agora::rtc::ILocalVideoTrack> videoTrack, const agora::rtc::LocalVideoTrackStats &stats) override {}
  void onAudioVolumeIndication(const agora::rtc::AudioVolumeInformation *speakers, unsigned int speakerNumber, int totalVolume) override {}
  void onLocalAudioTrackStatistics(const agora::rtc::LocalAudioStats &stats) override {}
  void onRemoteAudioTrackStatistics(agora::agora_refptr<agora::rtc::IRemoteAudioTrack> audioTrack, const agora::rtc::RemoteAudioTrackStats &stats) override {}
  void onUserInfoUpdated(agora::user_id_t userId, USER_MEDIA_INFO msg, bool val) override {}
  void onIntraRequestReceived() override {}
  void onAudioSubscribeStateChanged(const char *channel, agora::user_id_t uid, agora::rtc::STREAM_SUBSCRIBE_STATE oldState, agora::rtc::STREAM_SUBSCRIBE_STATE newState, int elapseSinceLastState) override {}
  void onVideoSubscribeStateChanged(const char *channel, agora::user_id_t uid, agora::rtc::STREAM_SUBSCRIBE_STATE oldState, agora::rtc::STREAM_SUBSCRIBE_STATE newState, int elapseSinceLastState) override {}
  void onAudioPublishStateChanged(const char *channel, agora::rtc::STREAM_PUBLISH_STATE oldState, agora::rtc::STREAM_PUBLISH_STATE newState, int elapseSinceLastState) override {}
  void onVideoPublishStateChanged(const char *channel, agora::rtc::STREAM_PUBLISH_STATE oldState, agora::rtc::STREAM_PUBLISH_STATE newState, int elapseSinceLastState) override {}
  void onFirstRemoteVideoFrameRendered(agora::user_id_t userId, int width, int height, int elapsed) override {}
  void onFirstRemoteVideoFrame(agora::user_id_t userId, int width, int height, int elapsed) override {}
  void onFirstRemoteAudioFrame(agora::user_id_t userId, int elapsed) override {}
  void onFirstRemoteAudioDecoded(agora::user_id_t userId, int elapsed) override {}
  void onFirstRemoteVideoDecoded(agora::user_id_t userId, int width, int height, int elapsed) override {}
  void onAudioTrackPublishStart(agora::agora_refptr<agora::rtc::ILocalAudioTrack> audioTrack) override {}
  void onAudioTrackUnpublished(agora::agora_refptr<agora::rtc::ILocalAudioTrack> audioTrack) override {}
  void onVideoTrackPublishStart(agora::agora_refptr<agora::rtc::ILocalVideoTrack> videoTrack) override {}
  void onVideoTrackUnpublished(agora::agora_refptr<agora::rtc::ILocalVideoTrack> videoTrack) override {}
  void onVideoSizeChanged(agora::user_id_t userId, int width, int height, int rotation) override {}
  void onActiveSpeaker(agora::user_id_t userId) override {}
  void onStreamMessage(agora::user_id_t userId, int streamId, const char *data, size_t length) override {}

  // --- IAudioFrameObserverBase overrides ---
  bool onRecordAudioFrame(const char *channelId, AudioFrame &audioFrame) override { return true; }
  bool onPlaybackAudioFrame(const char *channelId, AudioFrame &audioFrame) override;
  bool onMixedAudioFrame(const char *channelId, AudioFrame &audioFrame) override { return true; }
  bool onEarMonitoringAudioFrame(AudioFrame &audioFrame) override { return true; }
  bool onPlaybackAudioFrameBeforeMixing(const char *channelId, agora::media::base::user_id_t userId, AudioFrame &audioFrame) override;
  int getObservedAudioFramePosition() override { return agora::media::IAudioFrameObserverBase::AUDIO_FRAME_POSITION_PLAYBACK; }
  AudioParams getPlaybackAudioParams() override { return agora::media::IAudioFrameObserverBase::AudioParams(); }
  AudioParams getRecordAudioParams() override { return agora::media::IAudioFrameObserverBase::AudioParams(); }
  AudioParams getMixedAudioParams() override { return agora::media::IAudioFrameObserverBase::AudioParams(); }
  AudioParams getEarMonitoringAudioParams() override { return agora::media::IAudioFrameObserverBase::AudioParams(); }

  // --- IVideoFrameObserver2 overrides ---
  void onFrame(const char *channelId, agora::user_id_t remoteUid, const agora::media::base::VideoFrame *frame) override;

private:
  agora::rtc::ILocalUser *local_user_{nullptr};
  agora::agora_refptr<agora::rtc::IRemoteAudioTrack> remote_audio_track_;
  agora::agora_refptr<agora::rtc::IRemoteVideoTrack> remote_video_track_;
  TransceiverYuvPcm *transceiver_ = nullptr;
};

// --- TransceiverOptions Definition (BEFORE TransceiverYuvPcm) ---
struct TransceiverOptions
{
  std::string appId;
  std::string channelId;
  std::string userId;
  std::string token;
  std::string remoteUserId;
  struct
  {
    int sampleRate = DEFAULT_SAMPLE_RATE;
    int numOfChannels = DEFAULT_NUM_OF_CHANNELS;
  } audio;
  struct
  {
    int targetBitrate = DEFAULT_TARGET_BITRATE;
    int width = DEFAULT_VIDEO_WIDTH;
    int height = DEFAULT_VIDEO_HEIGHT;
    int frameRate = DEFAULT_FRAME_RATE;
    std::string streamType = STREAM_TYPE_HIGH;
  } video;
  // Whisper options
  bool useWhisper = true;            // Flag to enable Whisper
  std::string whisperModelPath = ""; // Default path
  // Llama options
  bool useLlama = true;            // Flag to enable Llama
  std::string llamaModelPath = ""; // Default path
  // Denoise options
  bool useDenoise = true;            // Flag to enable Denoise
  std::string denoiseModelPath = ""; // Default path
};


// --- TransceiverYuvPcm Class Definition ---
class TransceiverYuvPcm
{
private:
  agora::base::IAgoraService *service_;
  agora::agora_refptr<agora::rtc::IRtcConnection> connection_;
  std::shared_ptr<SampleConnectionObserver> connObserver_;
  SampleLocalUserObserver localUserObserver_;
  agora::agora_refptr<agora::rtc::IMediaNodeFactory> factory_;
  agora::agora_refptr<agora::rtc::IAudioPcmDataSender> audioPcmDataSender_;
  agora::agora_refptr<agora::rtc::IVideoFrameSender> videoFrameSender_;
  agora::agora_refptr<agora::rtc::ILocalAudioTrack> customAudioTrack_;
  agora::agora_refptr<agora::rtc::ILocalVideoTrack> customVideoTrack_;

  bool initialized_ = false;
  TransceiverOptions options_;
  std::vector<uint8_t> video_buffer_;

public:
  TransceiverYuvPcm() : localUserObserver_(this) {}
  ~TransceiverYuvPcm() { release(); }

  bool initialize(const TransceiverOptions &options);
  bool joinChannel();
  void echoAudioFrame(const agora::media::IAudioFrameObserver::AudioFrame &receivedFrame);
  void echoVideoFrame(const agora::media::IVideoFrameObserver::VideoFrame &receivedFrame);
  void release();
  void sendAudioFrame(const agora::media::IAudioFrameObserver::AudioFrame &frame);
  void sendGeneratedAudio(const uint16_t *buffer, size_t num_samples);

  std::unique_ptr<WhillatsTranscriber> _transcriber;
  // Whillats Members
  std::unique_ptr<WhillatsLlama> _llama;
  std::unique_ptr<WhillatsTTS> _tts;
};

// --- TTS Audio Callback (C-Style Static Function) ---
static void ttsAudioCallback(bool success, const uint16_t *buffer, size_t num_samples, void *user_data)
{
  if (success && buffer && num_samples > 0 && user_data)
  {
    TransceiverYuvPcm *transceiver = static_cast<TransceiverYuvPcm *>(user_data);
    AG_LOG(VERBOSE, "ttsAudioCallback: Received %zu audio samples.", num_samples);
    transceiver->sendGeneratedAudio(buffer, num_samples);
  }
  else if (!success)
  {
    AG_LOG(ERROR, "ttsAudioCallback reported an error.");
  }
}

// --- TransceiverYuvPcm Method Implementations (MOVED HERE) ---

bool TransceiverYuvPcm::initialize(const TransceiverOptions &options)
{
  options_ = options;
  service_ = createAndInitAgoraService(false, true, true);
  if (!service_)
  {
    AG_LOG(ERROR, "Failed to create Agora service!");
    return false;
  }

  agora::rtc::RtcConnectionConfiguration ccfg;
  ccfg.clientRoleType = agora::rtc::CLIENT_ROLE_BROADCASTER;
  ccfg.autoSubscribeAudio = true;
  ccfg.autoSubscribeVideo = true;
  ccfg.channelProfile = agora::CHANNEL_PROFILE_LIVE_BROADCASTING;
  connection_ = service_->createRtcConnection(ccfg);
  if (!connection_)
  {
    AG_LOG(ERROR, "Failed to create Agora connection!");
    release();
    return false;
  }

  connObserver_ = std::make_shared<SampleConnectionObserver>();
  connection_->registerObserver(connObserver_.get());

  agora::rtc::VideoSubscriptionOptions subscriptionOptions;
  if (options_.video.streamType == STREAM_TYPE_HIGH)
  {
    subscriptionOptions.type = agora::rtc::VIDEO_STREAM_HIGH;
  }
  else if (options_.video.streamType == STREAM_TYPE_LOW)
  {
    subscriptionOptions.type = agora::rtc::VIDEO_STREAM_LOW;
  }
  else
  {
    AG_LOG(ERROR, "Invalid stream type: %s", options_.video.streamType.c_str());
    release();
    return false;
  }
  if (options_.remoteUserId.empty())
  {
    AG_LOG(INFO, "Subscribe streams from all remote users");
    connection_->getLocalUser()->subscribeAllAudio();
    connection_->getLocalUser()->subscribeAllVideo(subscriptionOptions);
  }
  else
  {
    connection_->getLocalUser()->subscribeAudio(options_.remoteUserId.c_str());
    connection_->getLocalUser()->subscribeVideo(options_.remoteUserId.c_str(),
                                                subscriptionOptions);
  }

  auto local_user = connection_->getLocalUser();
  if (!local_user)
  {
    AG_LOG(ERROR, "Failed to get local user!");
    release();
    return false;
  }

  if (local_user->setPlaybackAudioFrameBeforeMixingParameters(
          options.audio.numOfChannels, options.audio.sampleRate))
  {
    AG_LOG(ERROR, "Failed to set playback audio frame before mixing parameters!");
    return false;
  }

  localUserObserver_.setLocalUser(local_user);

  int ret_audio_reg = local_user->registerAudioFrameObserver(&localUserObserver_);
  int ret_video_reg = local_user->registerVideoFrameObserver(&localUserObserver_);
  if (ret_audio_reg != 0)
  {
    AG_LOG(WARN, "Failed to register audio frame observer on local user: %d", ret_audio_reg);
  }
  if (ret_video_reg != 0)
  {
    AG_LOG(WARN, "Failed to register video frame observer on local user: %d", ret_video_reg);
  }

  factory_ = service_->createMediaNodeFactory();
  if (!factory_)
  {
    AG_LOG(ERROR, "Failed to create media node factory!");
    release();
    return false;
  }

  audioPcmDataSender_ = factory_->createAudioPcmDataSender();
  if (!audioPcmDataSender_)
  {
    AG_LOG(ERROR, "Failed to create audio data sender!");
    release();
    return false;
  }

  customAudioTrack_ = service_->createCustomAudioTrack(audioPcmDataSender_);
  if (!customAudioTrack_)
  {
    AG_LOG(ERROR, "Failed to create audio track!");
    release();
    return false;
  }

  videoFrameSender_ = factory_->createVideoFrameSender();
  if (!videoFrameSender_)
  {
    AG_LOG(ERROR, "Failed to create video frame sender!");
    release();
    return false;
  }

  customVideoTrack_ = service_->createCustomVideoTrack(videoFrameSender_);
  if (!customVideoTrack_)
  {
    AG_LOG(ERROR, "Failed to create video track!");
    release();
    return false;
  }

  agora::rtc::VideoEncoderConfiguration encoderConfig;
  encoderConfig.codecType = agora::rtc::VIDEO_CODEC_H264;
  encoderConfig.dimensions.width = options_.video.width;
  encoderConfig.dimensions.height = options_.video.height;
  encoderConfig.frameRate = options_.video.frameRate;
  encoderConfig.bitrate = options_.video.targetBitrate;
  customVideoTrack_->setVideoEncoderConfiguration(encoderConfig);

  initialized_ = true;
  AG_LOG(INFO, "Transceiver initialized successfully.");
  return true;
}

bool TransceiverYuvPcm::joinChannel()
{
  if (!initialized_)
  {
    AG_LOG(ERROR, "Transceiver not initialized.");
    return false;
  }

  int ret = connection_->connect(options_.token.c_str(), options_.channelId.c_str(),
                                 options_.userId.c_str());

  if (ret != 0)
  {
    AG_LOG(ERROR, "Failed to connect to Agora channel! Error code: %d", ret);
    return false;
  }
  AG_LOG(INFO, "Connecting to channel %s...", options_.channelId.c_str());

  auto local_user = connection_->getLocalUser();
  if (!local_user)
  {
    AG_LOG(ERROR, "Failed to get local user after connect!");
    connection_->disconnect();
    return false;
  }

  customAudioTrack_->setEnabled(true);
  ret = local_user->publishAudio(customAudioTrack_);
  if (ret != 0)
  {
    AG_LOG(ERROR, "Failed to publish audio track! Error code: %d", ret);
    connection_->disconnect();
    return false;
  }
  AG_LOG(INFO, "Published custom audio track.");

  customVideoTrack_->setEnabled(true);
  ret = local_user->publishVideo(customVideoTrack_);
  if (ret != 0)
  {
    AG_LOG(ERROR, "Failed to publish video track! Error code: %d", ret);
    local_user->unpublishAudio(customAudioTrack_);
    connection_->disconnect();
    return false;
  }
  AG_LOG(INFO, "Published custom video track.");

  return true;
}

void TransceiverYuvPcm::sendAudioFrame(const agora::media::IAudioFrameObserver::AudioFrame &frame)
{
    if (!audioPcmDataSender_ || !initialized_)
      return;

    int ret = audioPcmDataSender_->sendAudioPcmData(
        frame.buffer,
        frame.renderTimeMs,
        0, // ntp
        frame.samplesPerChannel,
        (agora::rtc::BYTES_PER_SAMPLE)frame.bytesPerSample,
        frame.channels,
        frame.samplesPerSec);

    if (ret != 0)
    {
      AG_LOG(WARN, "sendAudioPcmData failed: %d", ret);
    }
}

// Implementation for sending generated TTS audio
void TransceiverYuvPcm::sendGeneratedAudio(const uint16_t *buffer, size_t num_samples)
{
  if (!audioPcmDataSender_ || !initialized_ || !buffer || num_samples == 0)
    return;

  // Get sample rate from TTS (assuming it matches Agora's expected rate)
  // Assuming 1 channel based on typical TTS output
  int sampleRate = WhillatsTTS::getSampleRate();
  int channels = 1;
  agora::rtc::BYTES_PER_SAMPLE bytesPerSample = agora::rtc::TWO_BYTES_PER_SAMPLE; // uint16_t
  // size_t buffer_size_bytes = num_samples * sizeof(uint16_t); // Not needed for sendAudioPcmData

  AG_LOG(VERBOSE, "Sending generated audio: %zu samples, %d channels, %d Hz", num_samples, channels, sampleRate);

  agora::media::IAudioFrameObserver::AudioFrame frame;
  frame.buffer = (void*) buffer;
  frame.renderTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  frame.samplesPerChannel = num_samples;
  frame.bytesPerSample = bytesPerSample;
  frame.channels = channels;
  frame.samplesPerSec = sampleRate;
  sendAudioFrame(frame);
}

void TransceiverYuvPcm::echoAudioFrame(const agora::media::IAudioFrameObserver::AudioFrame &receivedFrame)
{
  if (!_transcriber) {
    sendAudioFrame(receivedFrame);
  }
  else
  {
    // Process audio buffer and send it later
    _transcriber->processAudioBuffer((uint8_t *)receivedFrame.buffer,
                                     receivedFrame.samplesPerChannel * receivedFrame.bytesPerSample);
  }
}

void TransceiverYuvPcm::echoVideoFrame(const agora::media::IVideoFrameObserver::VideoFrame &receivedFrame)
{
  if (!videoFrameSender_ || !initialized_)
    return;

  if (receivedFrame.yBuffer && receivedFrame.uBuffer && receivedFrame.vBuffer &&
      receivedFrame.height > 0 && receivedFrame.yStride > 0)
  {
    agora::media::base::ExternalVideoFrame videoFrame;
    videoFrame.type = agora::media::base::ExternalVideoFrame::VIDEO_BUFFER_RAW_DATA;
    videoFrame.format = agora::media::base::VIDEO_PIXEL_I420;
    videoFrame.stride = receivedFrame.yStride;
    videoFrame.height = receivedFrame.height;
    videoFrame.rotation = receivedFrame.rotation;
    videoFrame.timestamp = receivedFrame.renderTimeMs;
    videoFrame.cropLeft = 0;
    videoFrame.cropTop = 0;
    videoFrame.cropRight = 0;
    videoFrame.cropBottom = 0;

    size_t y_size = (size_t)receivedFrame.yStride * receivedFrame.height;
    size_t uv_stride = (receivedFrame.yStride + 1) / 2;
    size_t uv_height = (receivedFrame.height + 1) / 2;
    size_t u_size = uv_stride * uv_height;
    size_t v_size = uv_stride * uv_height;
    size_t total_size = y_size + u_size + v_size;

    try
    {
      video_buffer_.resize(total_size);
    }
    catch (const std::bad_alloc &e)
    {
      AG_LOG(ERROR, "Failed to allocate video buffer: %s", e.what());
      return;
    }

      memcpy(video_buffer_.data(), receivedFrame.yBuffer, y_size);
      memcpy(video_buffer_.data() + y_size, receivedFrame.uBuffer, u_size);
      memcpy(video_buffer_.data() + y_size + u_size, receivedFrame.vBuffer, v_size);

    videoFrame.buffer = video_buffer_.data();

    memset(videoFrame.matrix, 0, sizeof(videoFrame.matrix)); // Zero out the matrix
    if (receivedFrame.matrix)
    { // Copy if source matrix exists
      memcpy(videoFrame.matrix, receivedFrame.matrix, sizeof(videoFrame.matrix));
    }

    int ret = videoFrameSender_->sendVideoFrame(videoFrame);
    // if (ret != 0) { AG_LOG(WARN, "sendVideoFrame failed: %d", ret); }
  }
  else
  {
    AG_LOG(WARN, "Received video frame with missing YUV buffers or invalid dimensions. Cannot echo.");
  }
}

void TransceiverYuvPcm::release()
{
  if (!initialized_)
    return;
  initialized_ = false;

  AG_LOG(INFO, "Releasing transceiver resources...");

  // Stop Whillats components first
  if (_transcriber)
  {
    _transcriber->stop();
  }
  if (_tts)
  {
    _tts->stop();
  }
  if (_llama)
  {
    // Llama might have its own stop/release if needed, add here
    AG_LOG(INFO, "Stopping Llama...");
  }
  // Consider explicit destruction/reset if needed before service release
  _transcriber.reset();
  _tts.reset();
  _llama.reset(); // Ensure llama is reset

  // ... Unregister observers, unpublish, disconnect ...
  if (connection_)
  {
    auto local_user = connection_->getLocalUser();
    if (local_user)
    {
      AG_LOG(INFO, "Unregistering observers from local user...");
      local_user->unregisterAudioFrameObserver(&localUserObserver_);
      local_user->unregisterVideoFrameObserver(&localUserObserver_);
      AG_LOG(INFO, "Unpublishing tracks...");
      if (customAudioTrack_)
        local_user->unpublishAudio(customAudioTrack_);
      if (customVideoTrack_)
        local_user->unpublishVideo(customVideoTrack_);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    AG_LOG(INFO, "Disconnecting connection...");
    if (connObserver_)
      connection_->unregisterObserver(connObserver_.get());
    connection_->disconnect();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // ... Cleanup Agora resources ...
  connObserver_.reset();
  audioPcmDataSender_ = nullptr;
  videoFrameSender_ = nullptr;
  customAudioTrack_ = nullptr;
  customVideoTrack_ = nullptr;
  factory_ = nullptr;
  connection_ = nullptr;

  // Release service LAST
  if (service_)
  {
    AG_LOG(INFO, "Releasing Agora service...");
    service_->release();
    service_ = nullptr;
  }
  video_buffer_.clear();
  video_buffer_.shrink_to_fit();

  AG_LOG(INFO, "Transceiver released.");
}

// --- Embedded SampleLocalUserObserver Method Implementations (MOVED HERE) ---

SampleLocalUserObserver::~SampleLocalUserObserver()
{
  // Unregistration is handled in TransceiverYuvPcm::release on the local_user_
  // Remove this call to prevent potential use-after-free if local_user_ is invalid here
  // if (local_user_) {
  //     local_user_->unregisterLocalUserObserver(this);
  // }
}

void SampleLocalUserObserver::setLocalUser(agora::rtc::ILocalUser *user)
{
  if (local_user_)
  {
    local_user_->unregisterLocalUserObserver(this);
  }
  local_user_ = user;
  if (local_user_)
  {
    local_user_->registerLocalUserObserver(this);
  }
}

void SampleLocalUserObserver::onUserAudioTrackSubscribed(agora::user_id_t userId, agora::agora_refptr<agora::rtc::IRemoteAudioTrack> audioTrack)
{
  AG_LOG(INFO, "Remote user '%s' audio track subscribed.", userId ? userId : "<null>");
  if (!remote_audio_track_)
  {
    remote_audio_track_ = audioTrack;
    if (remote_audio_track_)
    {
      // Corrected: Use adjustPlayoutVolume
      int ret = remote_audio_track_->adjustPlayoutVolume(100);
      if (ret != 0)
      {
        AG_LOG(WARN, "Failed to adjust playout volume for user '%s': %d", userId ? userId : "<null>", ret);
      }
      else
      {
        AG_LOG(INFO, "Adjusted playout volume for user '%s'.", userId ? userId : "<null>");
      }
    }
  }
  else
  {
    AG_LOG(WARN, "Already observing audio from another user. Ignoring user '%s'.", userId ? userId : "<null>");
  }
}

void SampleLocalUserObserver::onUserVideoTrackSubscribed(agora::user_id_t userId, const agora::rtc::VideoTrackInfo &trackInfo, agora::agora_refptr<agora::rtc::IRemoteVideoTrack> videoTrack)
{
  AG_LOG(INFO, "Remote user '%s' video track subscribed.", userId ? userId : "<null>");
  if (!remote_video_track_)
  {
    remote_video_track_ = videoTrack;
    if (remote_video_track_)
    {
      AG_LOG(INFO, "Observing video track for user '%s'.", userId ? userId : "<null>");
    }
  }
  else
  {
    AG_LOG(WARN, "Already observing video from another user. Ignoring user '%s'.", userId ? userId : "<null>");
  }
}

bool SampleLocalUserObserver::onPlaybackAudioFrame(const char *channelId, AudioFrame &audioFrame)
{
  return true;
}

bool SampleLocalUserObserver::onPlaybackAudioFrameBeforeMixing(const char *channelId, agora::media::base::user_id_t userId, AudioFrame &audioFrame)
{
  if (transceiver_)
  {
    transceiver_->echoAudioFrame(audioFrame);
  }
  return true;
}

void SampleLocalUserObserver::onFrame(const char *channelId, agora::user_id_t remoteUid, const agora::media::base::VideoFrame *frame)
{
  if (transceiver_ && frame)
  {
    transceiver_->echoVideoFrame(*frame);
  }
}

// --- Main Function ---
static bool exitFlag = false;
static void SignalHandler(int sigNo)
{
  if (!exitFlag)
  { // Prevent multiple signals causing issues
    AG_LOG(INFO, "Received signal %d, initiating shutdown...", sigNo);
    exitFlag = true;
  }
}

// Modified whisper callback to optionally use Llama
static void whisperResponseCallback(bool success, const char *whisper_response, void *user_data)
{
  if (!success || !whisper_response || !user_data)
  {
    if (!success)
      AG_LOG(ERROR, "whisperResponseCallback reported an error.");
    return;
  }

  TransceiverYuvPcm *transceiver = static_cast<TransceiverYuvPcm *>(user_data);
  const char *text_to_speak = whisper_response; // Default to original whisper text
  std::string llama_response_str;

  // Check if Llama is enabled and initialized
  if (transceiver->_llama)
  {
    AG_LOG(INFO, "Sending text to Llama: '%s'", whisper_response);
    transceiver->_llama->askLlama(whisper_response);
  }
  else {
    // Send the chosen text to TTS
    if (transceiver->_tts)
    {
      AG_LOG(INFO, "Queueing text to TTS: '%s'", text_to_speak);
      transceiver->_tts->queueText(text_to_speak);
    }
    else
    {
      AG_LOG(WARN, "TTS object not available in whisperResponseCallback.");
    }
  }
}

// Modified whisper callback to optionally use Llama
static void llamaResponseCallback(bool success, const char *llama_response, void *user_data)
{
  if (!success || !llama_response || !user_data)
  {
    if (!success)
      AG_LOG(ERROR, "whisperResponseCallback reported an error.");
    return;
  }

  TransceiverYuvPcm *transceiver = static_cast<TransceiverYuvPcm *>(user_data);
  const char *text_to_speak = llama_response; // Default to original whisper text

  // Send the chosen text (either original or Llama's) to TTS
  if (transceiver->_tts)
  {
    AG_LOG(INFO, "Queueing text to TTS: '%s'", text_to_speak);
    transceiver->_tts->queueText(text_to_speak);    
  }
  else
  {
    AG_LOG(WARN, "TTS object not available in llamaResponseCallback.");
  }
}

int main(int argc, char *argv[])
{
  TransceiverOptions options;
  opt_parser optParser;

  optParser.add_long_opt("token", &options.token, "Agora token for authentication (must)");
  optParser.add_long_opt("appId", &options.appId, "Agora App ID (must)");
  optParser.add_long_opt("channelId", &options.channelId, "Channel name (must)");
  optParser.add_long_opt("userId", &options.userId, "User ID (string, optional, often in token)");
  optParser.add_long_opt("sampleRate", &options.audio.sampleRate, "Audio sample rate (e.g., 16000)");
  optParser.add_long_opt("numOfChannels", &options.audio.numOfChannels, "Audio channels (1 or 2)");
  optParser.add_long_opt("width", &options.video.width, "Video width (e.g., 352)");
  optParser.add_long_opt("height", &options.video.height, "Video height (e.g., 288)");
  optParser.add_long_opt("fps", &options.video.frameRate, "Video frame rate (e.g., 15)");
  optParser.add_long_opt("bitrate", &options.video.targetBitrate, "Video target bitrate in bps (e.g., 1000000)");
  optParser.add_long_opt("streamtype", &options.video.streamType, "the stream type");

  // Add Whisper options
  optParser.add_long_opt("useWhisper", &options.useWhisper, "Enable Whisper processing (flag)");
  optParser.add_long_opt("whisperModelPath", &options.whisperModelPath, "Path to Whisper model file");

  // Add Llama options
  optParser.add_long_opt("useLlama", &options.useLlama, "Enable Llama processing (flag)");
  optParser.add_long_opt("llamaModelPath", &options.llamaModelPath, "Path to Llama model file");

  // Add Denoise options
  optParser.add_long_opt("useDenoise", &options.useDenoise, "Enable Denoise processing (flag)");
  optParser.add_long_opt("denoiseModelPath", &options.denoiseModelPath, "Path to Denoise model file");

  if ((argc <= 1) || !optParser.parse_opts(argc, argv))
  {
    std::ostringstream strStream;
    optParser.print_usage(argv[0], strStream);
    std::cout << strStream.str() << std::endl;
    return -1;
  }

  if (options.token.empty())
  {
    AG_LOG(ERROR, "--token is required!");
    return -1;
  }
  if (options.channelId.empty())
  {
    AG_LOG(ERROR, "--channelId is required!");
    return -1;
  }

  // Setup signal handling
  std::signal(SIGQUIT, SignalHandler);
  std::signal(SIGABRT, SignalHandler);
  std::signal(SIGINT, SignalHandler);

  AG_LOG(INFO, "Initializing Transceiver...");
  std::unique_ptr<TransceiverYuvPcm> transceiver = std::make_unique<TransceiverYuvPcm>();

  if (!transceiver->initialize(options))
  {
    AG_LOG(ERROR, "Failed to initialize transceiver!");
    return -1;
  }

  // Initialize callbacks
  WhillatsSetResponseCallback whisper_callback(whisperResponseCallback, transceiver.get());
  WhillatsSetAudioCallback tts_callback(ttsAudioCallback, transceiver.get());
  WhillatsSetResponseCallback llama_callback(llamaResponseCallback, transceiver.get());

  // Initialize Whillats Transcriber
  if (options.useWhisper)
  {
    transceiver->_transcriber = std::make_unique<WhillatsTranscriber>(options.whisperModelPath.c_str(),
                                                                      whisper_callback);
    if (!transceiver->_transcriber->start())
    {
      AG_LOG(ERROR, "Failed to start WhillatsTranscriber!");
      return -1;
    }
    AG_LOG(INFO, "WhillatsTranscriber started.");
  }
  else
  {
    AG_LOG(INFO, "Whisper processing is disabled.");
  }

  // Initialize Whillats TTS
  transceiver->_tts = std::make_unique<WhillatsTTS>(tts_callback);
  if (!transceiver->_tts->start())
  {
    AG_LOG(ERROR, "Failed to start WhillatsTTS!");
    // Consider cleanup before exiting
    return -1;
  }
  AG_LOG(INFO, "WhillatsTTS started.");

  // Initialize Whillats Llama *conditionally*
  if (options.useLlama)
  {
    AG_LOG(INFO, "Initializing WhillatsLlama with model: %s", options.llamaModelPath.c_str());
    // Assuming constructor takes only the model path
    transceiver->_llama = std::make_unique<WhillatsLlama>(
        options.llamaModelPath.c_str(),
        llama_callback);

    if (!transceiver->_llama->start())
    {
      AG_LOG(ERROR, "Failed to start WhillatsLlama!");
      return -1;
    }
    AG_LOG(INFO, "WhillatsLlama started successfully.");
  }
  else
  {
    AG_LOG(INFO, "Llama processing is disabled.");
  }

  AG_LOG(INFO, "Joining channel...");
  if (!transceiver->joinChannel())
  {
    AG_LOG(ERROR, "Failed to join channel!");
    return -1;
  }

  AG_LOG(INFO, "Transceiver running. Press Ctrl+C or send SIGINT/SIGQUIT to exit.");

  while (!exitFlag)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  AG_LOG(INFO, "Exiting application...");
  // transceiver->release() will be called automatically by unique_ptr destructor
  AG_LOG(INFO, "Application finished.");
  return 0;
}