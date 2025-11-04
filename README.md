# College-Face-Attendance-System

Face Registration of student using a camera and the captured face images are stored.
From the saved face images, face descriptor using dlib library for each person is generated and stored in a csv file.
Face recognition is done via comparing the face descriptors of the currently captured face with the stored face descriptors and if they match the person is verified.
Using SQlite, a database is created where attendance records are stored.
Web Interface is used for accessing attendance data on the basis of date and time.
