from pickle import NONE
import sys
import sqlite3
from detect import detect_emotion
from stt import analyze_tone

#print(str(len(sys.argv)))

connection = sqlite3.connect('customers.db')
cursor = connection.cursor()




if __name__ == "__main__":
   total_score = 0
   if(len(sys.argv) == 4):
      id = sys.argv[3]
      for x in range(1,int(sys.argv[2])+1):
         print("samples\\" + sys.argv[1]+"0"+str(x)+".wav")
         emotion_score = detect_emotion("samples\\" + sys.argv[1]+"0"+str(x)+".wav")
         context_score = analyze_tone("samples\\" + sys.argv[1]+"0"+str(x)+".wav")
         print(context_score)
         print(emotion_score)
         if(emotion_score == -1 and context_score == -1):
            total_score = total_score -1
         elif((emotion_score == -1 and context_score == 0) or(emotion_score == 0 and context_score == -1)):
            total_score = total_score -0.6
         elif(emotion_score == 1 and context_score == 1):
            total_score = total_score +1
         elif((emotion_score == 1 and context_score == 0) or(emotion_score == 0 and context_score == 1)):
            total_score = total_score +0.6
         

      if(total_score > 5):
         total_score = 5
      elif(total_score <-5):
         total_score = -5
      print( "total score: " + str(total_score))
      
   
      query = ('''SELECT contentlevel FROM customers WHERE id = ?''')
      result = cursor.execute(query,id)
      connection.commit()
      data = cursor.fetchone()
   
      if data:
         query = """Update customers set contentlevel = ? where id = ?"""
         last_score = float(data[0])
         print("last score: " + str(last_score))
         total_score = (last_score + (total_score*1.2))/2
         if(total_score > 5):
            total_score = 5
         elif(total_score <-5):
            total_score = -5
         print("total score: " + str(total_score))
      
      else:
         query = '''INSERT INTO customers(contentlevel,id)
         VALUES(?,?)'''

      data = (total_score,id)
      cursor.execute(query,data)
      connection.commit()

   else:
      print("illegal call, make sure you call the app with proper arguments")

   connection.close()


   

