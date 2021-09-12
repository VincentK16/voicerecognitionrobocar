#include"TFT_eSPI.h"
#include <project_47606_inferencing.h>
enum {ADC_BUF_LEN = 1600};
typedef struct {
    uint16_t btctrl;
    uint16_t btcnt;
    uint32_t srcaddr;
    uint32_t dstaddr;
    uint32_t descaddr;
}dmacdescriptor;
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
}inference_t;
volatile uint8_t recording = 0;
uint16_t adc_buf_0[ADC_BUF_LEN];
uint16_t adc_buf_1[ADC_BUF_LEN];
volatile dmacdescriptor wrb[DMAC_CH_NUM] __attribute__ ((aligned (16)));
dmacdescriptor descriptor_section[DMAC_CH_NUM] __attribute__ ((aligned (16)));
dmacdescriptor descriptor __attribute__ ((aligned (16)));
static inference_t inference;

class FilterBuHp1{
    public:
    FilterBuHp1(){
        v[0] = 0.0;
    }
    private:
    float v[2];
    public:
    float step(float x)
    {
        v[0] = v[1];
        v[1] = (9.621952458291035404e-1f * x)
                + (0.92439049165820696974f * v[0]);
        return
        (v[1] - v[0]);
    }
};
FilterBuHp1 filter;

static void audio_rec_callback(uint16_t *buf, uint32_t buf_len) {
    if (recording) {
        for (uint32_t i = 0; i < buf_len; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = filter.step(((int16_t)buf[i] - 1024) * 16);
            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

void DMAC_1_Handler() {
    static uint8_t count = 0;
    if (DMAC->Channel[1].CHINTFLAG.bit.SUSP) {
        DMAC->Channel[1].CHCTRLB.reg = DMAC_CHCTRLB_CMD_RESUME;
        DMAC->Channel[1].CHINTFLAG.bit.SUSP = 1;
        if (count) {
            audio_rec_callback(adc_buf_0, ADC_BUF_LEN);
        }else {
            audio_rec_callback(adc_buf_1, ADC_BUF_LEN);
        }
        count = (count + 1) % 2;
    }
}

void config_dma_adc() {
    DMAC->BASEADDR.reg = (uint32_t)descriptor_section;
    DMAC->WRBADDR.reg = (uint32_t)wrb;
    DMAC->CTRL.reg = DMAC_CTRL_DMAENABLE | DMAC_CTRL_LVLEN(0xf);
    DMAC->Channel[1].CHCTRLA.reg = DMAC_CHCTRLA_TRIGSRC(TC5_DMAC_ID_OVF) |
                                    DMAC_CHCTRLA_TRIGACT_BURST;

    descriptor.descaddr = (uint32_t)&descriptor_section[1];
    descriptor.srcaddr = (uint32_t)&ADC1->RESULT.reg;
    descriptor.dstaddr = (uint32_t)adc_buf_0 + sizeof(uint16_t) * ADC_BUF_LEN;
    descriptor.btcnt = ADC_BUF_LEN;
    descriptor.btctrl = DMAC_BTCTRL_BEATSIZE_HWORD |
                        DMAC_BTCTRL_DSTINC |
                        DMAC_BTCTRL_VALID |
                        DMAC_BTCTRL_BLOCKACT_SUSPEND;
    memcpy(&descriptor_section[0], &descriptor, sizeof(descriptor));

    descriptor.descaddr = (uint32_t)&descriptor_section[0];
    descriptor.srcaddr = (uint32_t)&ADC1->RESULT.reg;
    descriptor.dstaddr = (uint32_t)adc_buf_1 + sizeof(uint16_t) * ADC_BUF_LEN;
    descriptor.btcnt = ADC_BUF_LEN;
    descriptor.btctrl = DMAC_BTCTRL_BEATSIZE_HWORD |
                        DMAC_BTCTRL_DSTINC |
                        DMAC_BTCTRL_VALID |
                        DMAC_BTCTRL_BLOCKACT_SUSPEND;
    memcpy(&descriptor_section[1], &descriptor, sizeof(descriptor));

    NVIC_SetPriority(DMAC_1_IRQn, 0);
    NVIC_EnableIRQ(DMAC_1_IRQn);

    DMAC->Channel[1].CHINTENSET.reg = DMAC_CHINTENSET_SUSP;

    ADC1->INPUTCTRL.bit.MUXPOS = ADC_INPUTCTRL_MUXPOS_AIN12_Val;
    while (ADC1->SYNCBUSY.bit.INPUTCTRL);
    ADC1->SAMPCTRL.bit.SAMPLEN = 0x00;
    while (ADC1->SYNCBUSY.bit.SAMPCTRL);
    ADC1->CTRLA.reg = ADC_CTRLA_PRESCALER_DIV128;
    ADC1->CTRLB.reg = ADC_CTRLB_RESSEL_12BIT |
                    ADC_CTRLB_FREERUN;
    while (ADC1->SYNCBUSY.bit.CTRLB);
    ADC1->CTRLA.bit.ENABLE = 1;
    while (ADC1->SYNCBUSY.bit.ENABLE);
    ADC1->SWTRIG.bit.START = 1;
    while (ADC1->SYNCBUSY.bit.SWTRIG);
    DMAC->Channel[1].CHCTRLA.bit.ENABLE = 1;
    GCLK->PCHCTRL[TC5_GCLK_ID].reg = GCLK_PCHCTRL_CHEN |
                                    GCLK_PCHCTRL_GEN_GCLK1;
    TC5->COUNT16.WAVE.reg = TC_WAVE_WAVEGEN_MFRQ;
    TC5->COUNT16.CC[0].reg = 3000 - 1;
    while (TC5->COUNT16.SYNCBUSY.bit.CC0);
    TC5->COUNT16.CTRLA.bit.ENABLE = 1;
    while (TC5->COUNT16.SYNCBUSY.bit.ENABLE);
}

static bool microphone_inference_record(void) {
    bool ret = true;
    while (inference.buf_ready == 0) {
        delay(1);
    }
    inference.buf_ready = 0;
    return ret;
}

static int microphone_audio_signal_get_data(size_t offset,
                                              size_t length,
                                              float *out_ptr) {
      numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);
      return 0;
}

TFT_eSPI tft;
ei_impulse_result_classification_t currentClassification[EI_CLASSIFIER_LABEL_COUNT];
const char* maxConfidenceLabel;

void runClassifier()
{
    bool m = microphone_inference_record();
    if (!m) {
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = { 0 };
    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, false);
    if (r != EI_IMPULSE_OK) {
        return;
    }

    float maxValue = 0;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_impulse_result_classification_t classification_t = result.classification[ix];
        ei_printf("    %s: %.5f\n", classification_t.label, classification_t.value);
        float value = classification_t.value;
        if (value > maxValue) {
            maxValue = value;
            maxConfidenceLabel = classification_t.label;
        }
        currentClassification[ix] = classification_t;
    }
}
float getLabelConfidence(char *label)
{
    float value = 0;
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
        ei_impulse_result_classification_t classification = currentClassification[ix];
        if (label == classification.label) {
            value = classification.value;
        }
    }
    return value;
}

void setup(){
  Serial1.begin(9600);
  tft.begin();
    run_classifier_init();
    inference.buffers[0] = (int16_t *)malloc(EI_CLASSIFIER_SLICE_SIZE * sizeof(int16_t));
    if (inference.buffers[0] == NULL) {
        return;
    }
    inference.buffers[1] = (int16_t *)malloc(EI_CLASSIFIER_SLICE_SIZE * sizeof(int16_t));
    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = EI_CLASSIFIER_SLICE_SIZE;
    inference.buf_ready = 0;

    config_dma_adc();

    recording = 1;

  tft.setRotation(3);
  tft.setTextSize(3);

}



void loop(){

  runClassifier();
  tft.fillScreen(0x7E1);
  tft.drawString((String)maxConfidenceLabel, 130, 50);
  if ((getLabelConfidence("go") > 0.8)) {
    Serial1.println('1');
    delay(1000);
  } else {
    if ((getLabelConfidence("stop") > 0.8)) {
      Serial1.println('2');
      delay(1000);
    } else {
      Serial1.println('0');
      delay(1000);
    }
  }

}
