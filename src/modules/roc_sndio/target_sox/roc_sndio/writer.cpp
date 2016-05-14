/*
 * Copyright (c) 2015 Mikhail Baranov
 * Copyright (c) 2015 Victor Gaydov
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <vector>

#include "roc_core/scoped_ptr.h"
#include "roc_core/panic.h"
#include "roc_core/log.h"

#include "roc_sndio/writer.h"
#include "roc_sndio/default.h"
#include "roc_sndio/init.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#include <alsa/asoundlib.h>

#define oops(func) (fprintf(stderr, "%s\n", func), exit(1))

namespace roc {
namespace sndio {

static const int n_channels = 2, sample_rate = 44100;

static void set_hw_params(snd_pcm_t* pcm,
                   snd_pcm_uframes_t* period_size, snd_pcm_uframes_t* buffer_size) {
    //
    snd_pcm_hw_params_t* hw_params = NULL;
    snd_pcm_hw_params_alloca(&hw_params);

    // initialize hw_params
    if (snd_pcm_hw_params_any(pcm, hw_params) < 0) {
        oops("snd_pcm_hw_params_any");
    }

    // enable software resampling
    if (snd_pcm_hw_params_set_rate_resample(pcm, hw_params, 1) < 0) {
        oops("snd_pcm_hw_params_set_rate_resample");
    }

    // set number of channels
    if (snd_pcm_hw_params_set_channels(pcm, hw_params, n_channels) < 0) {
        oops("snd_pcm_hw_params_set_channels");
    }

    // set interleaved format (L R L R ...)
    if (int ret =
        snd_pcm_hw_params_set_access(pcm, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        oops("snd_pcm_hw_params_set_access");
    }

    // set little endian 32-bit floats
    if (snd_pcm_hw_params_set_format(pcm, hw_params, SND_PCM_FORMAT_FLOAT_LE) < 0) {
        oops("snd_pcm_hw_params_set_format");
    }

    // set sample rate
    unsigned int rate = sample_rate;
    if (snd_pcm_hw_params_set_rate_near(pcm, hw_params, &rate, 0) < 0) {
        oops("snd_pcm_hw_params_set_rate_near");
    }
    if (rate != sample_rate) {
        oops("can't set sample rate (exact value is not supported)");
    }

    // set period time in microseconds
    // ALSA reads 'period_size' samples from circular buffer every period
    unsigned int period_time = sample_rate / 4;
    if (int ret =
        snd_pcm_hw_params_set_period_time_near(pcm, hw_params, &period_time, NULL) < 0) {
        oops("snd_pcm_hw_params_set_period_time_near");
    }

    // get period size, i.e. number of samples fetched from circular buffer
    // every period, calculated from 'sample_rate' and 'period_time'
    *period_size = 0;
    if (snd_pcm_hw_params_get_period_size(hw_params, period_size, NULL) < 0) {
        oops("snd_pcm_hw_params_get_period_size");
    }

    // set buffer size, i.e. number of samples in circular buffer
    *buffer_size = *period_size * 8;
    if (snd_pcm_hw_params_set_buffer_size_near(pcm, hw_params, buffer_size) < 0) {
        oops("snd_pcm_hw_params_set_buffer_size_near");
    }

    // get buffer time, i.e. total duration of circular buffer in microseconds,
    // calculated from 'sample_rate' and 'buffer_size'
    unsigned int buffer_time = 0;
    if (snd_pcm_hw_params_get_buffer_time(hw_params, &buffer_time, NULL) < 0) {
        oops("snd_pcm_hw_params_get_buffer_time");
    }

    printf("period_size = %ld\n", (long)*period_size);
    printf("period_time = %ld\n", (long)period_time);
    printf("buffer_size = %ld\n", (long)*buffer_size);
    printf("buffer_time = %ld\n", (long)buffer_time);

    // send hw_params to ALSA
    if (snd_pcm_hw_params(pcm, hw_params) < 0) {
        oops("snd_pcm_hw_params");
    }
}

static void set_sw_params(snd_pcm_t* pcm,
                   snd_pcm_uframes_t period_size, snd_pcm_uframes_t buffer_size) {
    //
    snd_pcm_sw_params_t* sw_params = NULL;
    snd_pcm_sw_params_alloca(&sw_params);

    // initialize sw_params
    if (snd_pcm_sw_params_current(pcm, sw_params) < 0) {
        oops("snd_pcm_sw_params_current");
    }

    // set start threshold to buffer_size, so that ALSA starts playback only
    // after circular buffer becomes full first time
    if (snd_pcm_sw_params_set_start_threshold(pcm, sw_params, buffer_size) < 0) {
        oops("snd_pcm_sw_params_set_start_threshold");
    }

    // set minimum number of samples that can be read by ALSA, so that it'll
    // wait until there are at least 'period_size' samples in circular buffer
    if (snd_pcm_sw_params_set_avail_min(pcm, sw_params, period_size) < 0) {
        oops("snd_pcm_sw_params_set_avail_min");
    }

    // send sw_params to ALSA
    if (snd_pcm_sw_params(pcm, sw_params) < 0) {
        oops("snd_pcm_sw_params");
    }
}

static void Run(audio::ISampleBufferReader& input) {
    snd_pcm_t* pcm = NULL;
    if (snd_pcm_open(&pcm, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
        oops("snd_pcm_open");
    }

    snd_pcm_uframes_t period_size = 0, buffer_size = 0;
    set_hw_params(pcm, &period_size, &buffer_size);
    set_sw_params(pcm, period_size, buffer_size);

    const int buf_sz = period_size * n_channels * sizeof(float);
    float* buf = (float*)malloc(buf_sz);

    std::vector<float> tmp;

    for (;;) {
        while (tmp.size() < period_size * n_channels) {
            audio::ISampleBufferConstSlice b = input.read();
            if (!b) {
                goto out;
            }

            const size_t ts = tmp.size();

            tmp.resize(tmp.size() + b.size());

            memcpy(&tmp[0] + ts, b.data(), b.size() * sizeof(float));
        }

        size_t s = tmp.size();
        if (s > (period_size * n_channels)) {
            s = (period_size * n_channels);
        }

        memcpy(buf, &tmp[0], s * sizeof(float));

        tmp.erase(tmp.begin(), tmp.begin() + s);

        int ret = snd_pcm_writei(pcm, buf, period_size);

        if (ret < 0) {
            if ((ret = snd_pcm_recover(pcm, ret, 1)) == 0) {
                printf("recovered after xrun (overrun/underrun)\n");
            }
        }

        if (ret < 0) {
            oops("snd_pcm_writei");
        }
    }

out:

    free(buf);

    snd_pcm_drain(pcm);
    snd_pcm_close(pcm);
}


Writer::Writer(audio::ISampleBufferReader& input,
               packet::channel_mask_t channels,
               size_t sample_rate)
    : input_(input) {
}

Writer::~Writer() {
}

bool Writer::open(const char*, const char*) {
    return true;
}

void Writer::stop() {
}

void Writer::run() {
    Run(input_);
}

} // namespace sndio
} // namespace roc
