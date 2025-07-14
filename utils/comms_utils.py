import socket
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from pypylon import pylon

#region Kamera

# Try to connect to first camera found and return camera object if successfull
def camera_connect():
    camera = None
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        if camera.IsGrabbing():
            camera.StopGrabbing()
    except:
        print("Error - Couldn't connect to camera")
    return camera

# Set key parameters to given camera with given data from dataclass CalibrationData
def camera_set_parameters(camera, data):
    camera.ExposureTime.Value = data.exposure_time
    camera.AcquisitionFrameRateEnable.Value = True
    camera.AcquisitionFrameRate.Value = data.framerate
    camera.DeviceLinkThroughputLimitMode.Value = "Off"
    camera.GainSelector.Value = "All"
    camera.Gain.Value = data.gain
    camera.Width.SetValue(data.camera_size[0])
    camera.Height.SetValue(data.camera_size[1])
    camera.OffsetX.SetValue(data.camera_offset[0])
    camera.OffsetY.SetValue(data.camera_offset[1])
    camera.MaxNumBuffer.Value = 10
    camera.PixelFormat.Value = "BGR8"
    print(f"Resulting Camera-Framerate: {camera.BslResultingAcquisitionFrameRate.Value}")



#endregion


#region UDP

# Initialise UDP Socket with ip and port
def udp_init(ip: str = "127.0.0.1", port: int = 1904):
    udp_socket, udp_address
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.settimeout(5.0)
    udp_address = (ip, port)
    return udp_socket, udp_address

# Close UDP Socket if connected
def udp_close(udp_socket, udp_address):
    if udp_socket:
        udp_socket.close()
    udp_socket = None
    udp_address = None


# Send Data as struct with specified format, returns sequencenumber+1
def udp_send_data_struct(udp_socket, udp_address, info, seq: int, format: str = "iff", ):
    data = struct.pack(format, seq, *info)
    udp_socket.sendto(data, udp_address)
    return seq+1

# Receive Data with given buffer size, return data or None in case of timeout
def udp_receive_data(udp_socket, buffer_size: int = 1024):
    try:
        data, addr = udp_socket.recvfrom(buffer_size)
        return data
    except socket.timeout:
        return None

# Perform Ping test with given num_pings, randomized info and given format
def udp_ping_test(udp_socket, udp_address, num_pings: int = 1000, info = [1.0, 2.0], format: str = "iff"):
    lost_packages = 0 # For Count of lost packages
    times = [] # For Round-Trip-Times
    seq = 0 # Initial sequence-number

    num_floats = len(format)-1

    for i in range(num_pings):
        info = list(np.random.uniform(-1.0 , 1.0, size=num_floats)) # Randomize Data between -1 and 1

        send_time = time.perf_counter() # Start Timer 
        seq = udp_send_data_struct(udp_socket, udp_address, info, seq, format) # Send Data
        received = udp_receive_data(udp_socket=udp_socket) # Receive Data
        received_time = time.perf_counter() #Stop Timer

        # Check return for
        if received is None:
            lost_packages += 1
            print("Lost because timeout")
        else:
            received = struct.unpack(format, received)
            received_seq = received[0]
            received_data = received[1:]
            if received_seq == seq-1 and np.allclose(received_data, info, atol=1e-8): # Compares Sequence-Number and Data with Tolerance of 1e-8
                times.append((received_time-send_time)*1000)
            else:
                lost_packages += 1
                print(f"Lost because wrong feedback, {seq-1} vs. {received_seq} | {info} vs {received_data}")

    times = np.array(times, dtype=np.float32)
    avg = np.mean(times)
    print(f"Lost packages: {lost_packages}")
    print(f"Average Time: {avg}")
    return times, lost_packages

# Perform series of pings tests
def udp_statistics(num_pings: int = 100000, min_floats: int = 2, max_floats: int = 12):
    
    all_times = []
    labels = []
    
    # Perform ping-test for different formats
    for p in range(min_floats, max_floats+1,2):
        format = "i" + "f" * p
        labels.append(format)

        udp_socket, udp_address = udp_init()
        times, numlost =  udp_ping_test(udp_socket, udp_address, 1000, format=format)
        udp_close(udp_socket, udp_address)
        all_times.append(times)

    # Plot results as boxplot
    plt.figure(figsize=(12,6))
    plt.boxplot(all_times, tick_labels=labels, showfliers=False, vert=True, patch_artist=True, showmeans=True, meanline=True)
    plt.text(
        0.5,                        
        0.95,                       
        f"Number of packets per format: {num_pings:.1e}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='gray')
    )
    plt.xlabel("Data-Format ")
    plt.ylabel("Roundtrip-Time in ms")
    plt.grid(True, axis='y')
    plt.show()

    # Plot results as boxplot
    plt.figure(figsize=(12,6))
    plt.boxplot(all_times, tick_labels=labels, showfliers=True, vert=True, patch_artist=True, showmeans=True, meanline=True)
    plt.text(
        0.5,                        
        0.95,                       
        f"Number of packets per format: {num_pings:.1e}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='gray')
    )
    plt.xlabel("Data-Format ")
    plt.ylabel("Roundtrip-Time in ms")
    plt.grid(True, axis='y')
    plt.show()


#endregion



#region TCP
tcp_socket = None

def tcp_init(ip: str = "127.0.0.1", port: int = 1904):
    global tcp_socket
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((ip, port))

#endregion




if __name__ == "__main__":
    udp_statistics(num_pings=1000000000, min_floats=2, max_floats=12)

