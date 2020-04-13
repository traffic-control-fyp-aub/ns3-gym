#ifndef TRAFFIC_INFO_SERVER_H
#define TRAFFIC_INFO_SERVER_H

#include "ns3/traci-client.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/address.h"
#include "ns3/ipv4-address.h"
#include "ns3/traced-callback.h"
#include "ns3/rsu-environment.h"

namespace ns3 {

std::vector<std::string> split (const std::string &input, const std::string &regex);

class Socket;
class Packet;

/**
	 * \ingroup applications 
	 * \defgroup TrafficInfo TrafficInfo
	 */

/**
 * \ingroup TrafficInfo
	 * \brief A structure to hold vehicle related values
	 *
	 * THis structure saves vehicle related parameters such as speeds , headways and other emission related data.
 */
struct vehicel_data
{

  std::string vehicle_id; // id of vehicle
  double speed; // vehicle speed m/s
  double headway; // time to reach leading vehicle in s
  int lane_index; // index of lane within road [1,2,..]
  double fuel_consumption; // consumption of fuel
  double emission_co2; // emmission of carbon dioxide
  double emission_co; // emission of carbon monoxide
  double emission_nox; // emission of nitrogen oxides
  double emission_pmx; // emission of particulate matter

  vehicel_data (std::string _vehicle_id, double _speed, double _headway, int _lane_index,
                double _fuel_consumption, double _emission_co2, double _emission_co,
                double _emission_nox, double _emission_pmx)
  {
    vehicle_id = _vehicle_id;
    speed = _speed;
    headway = _headway;
    lane_index = _lane_index;
    fuel_consumption = _fuel_consumption;
    emission_co2 = _emission_co2;
    emission_co = _emission_co;
    emission_nox = _emission_nox;
    emission_pmx = _emission_pmx;
  }
};

/**
	 * \ingroup TrafficInfo
	 * \brief A Traffic Info server
	 *
	 * Traffic information is broadcasted
	 */
class RsuSpeedControl : public Application
{
public:
  /**
		 * \brief Get the type ID.
		 * \return the object TypeId
		 */
  static TypeId GetTypeId (void);
  RsuSpeedControl ();
  virtual ~RsuSpeedControl ();
  Ptr<RsuEnv> GetEnv ();

protected:
  virtual void DoDispose (void);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  /**
		 * \brief Schedule the next packet transmission
		 * \param dt time interval between packets.
		 */
  void ScheduleTransmit (Time dt);

  /**
		 * \brief Send a packet
		 */
  void Send (void);
  void ChangeSpeed (void);
  void ClearDataTable (void);
  void HandleRead (Ptr<Socket> socket);

  uint16_t m_port; //!< Port on which traffic information is sent
  Time m_interval; //!< Packet inter-send time
  Time m_clear_interval; //!< Packet inter-send time
  uint32_t m_count; //!< Maximum number of packets the application will send
  Ptr<Socket> tx_socket; //!< IPv4 Socket
  Ptr<Socket> rx_socket; //!< IPv4 Socket
  EventId m_sendEvent; //!< Event to send the next packet
  Ptr<TraciClient> m_client;
  std::map<std::string, std::pair<double, double>> m_vehicles_data;

  /// Callbacks for tracing the packet Tx events
  TracedCallback<Ptr<const Packet>> m_txTrace;

  // GymEnv
  Ptr<RsuEnv> m_rsu_gym_env; //!< Gym environment object
};

class VehicleSpeedControl : public Application
{
public:
  /**
		 * \brief Get the type ID.
		 * \return the object TypeId
		 */
  static TypeId GetTypeId (void);

  VehicleSpeedControl ();

  virtual ~VehicleSpeedControl ();

  void StopApplicationNow ();

protected:
  virtual void DoDispose (void);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  /**
		 * \brief Handle a packet reception.
		 *
		 * This function is called by lower layers.
		 *
		 * \param socket the socket the packet was received to.
		 */
  void HandleRead (Ptr<Socket> socket);
  void Send (void);
  void ScheduleTransmit (Time dt);

  Ptr<Socket> rx_socket; //!< Socket
  uint16_t m_port; //!< Port on which client will listen for traffic information
  EventId m_sendEvent; //!< Event to send the next packet
  Time m_interval; //!< Packet inter-send time
  Ptr<Socket> tx_socket; //!< Socket
  Ptr<TraciClient> m_client;
  double last_velocity;
  double last_headway;
};

} // namespace ns3

#endif /* TRAFFIC_INFO_SERVER_H */
