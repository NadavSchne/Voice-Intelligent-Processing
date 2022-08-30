from ibm_watson import SpeechToTextV1, ToneAnalyzerV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, authenticator




tone_analyzer_apikey = 
tone_analyzer_url = 'https://api.au-syd.tone-analyzer.watson.cloud.ibm.com/instances/f98a8a4b-df9a-4800-8f99-201f4187433b'
STT_apikey =
STT_url = 'https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/7ce2ff91-f6ce-4641-b346-0554ab834970'


authenticator = IAMAuthenticator(tone_analyzer_apikey)
authenticator2 = IAMAuthenticator(STT_apikey)

ta = ToneAnalyzerV3(version="2017-09-21", authenticator=authenticator)
ta.set_service_url(tone_analyzer_url)
stt = SpeechToTextV1(authenticator=authenticator2)
stt.set_service_url(STT_url)

def analyze_tone(filename):
   with open(filename,'rb') as f:
      STT_result = stt.recognize(audio=f,content_type='audio/wav',model='en-US_NarrowbandModel').get_result()

   STT_confidence = STT_result['results'][0]['alternatives'][0]['confidence']
   STT_transcript = STT_result['results'][0]['alternatives'][0]['transcript']
   print(STT_transcript)
   print('stt confidence level:', STT_confidence)

   TA_result = ta.tone(STT_transcript).get_result()

   if(TA_result['document_tone']['tones']):
      TA_emotion = TA_result['document_tone']['tones'][0]['tone_id']
      print(TA_emotion)
      if(STT_confidence > 0.46):
         if(TA_emotion == 'joy'):
            return 1
         if(TA_emotion == 'anger'):
            return -1
         if(TA_emotion == 'sadness'):
            return -1
         return 0
   else:
      return 0



