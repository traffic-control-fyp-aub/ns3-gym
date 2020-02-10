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

namespace ns3 {

std::vector<std::string> getPacketData(const std::string& input, const std::string& regex);

class Socket;
class Packet;

/**
 * \ingroup applications 
 * \defgroup TrafficInfo TrafficInfo
 */

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
  void HandleRead (Ptr<Socket> socket);

  uint16_t m_port; //!< Port on which traffic information is sent
  Time m_interval; //!< Packet inter-send time
  uint32_t m_count; //!< Maximum number of packets the application will send
  Ptr<Socket> tx_socket; //!< IPv4 Socket
  Ptr<Socket> rx_socket; //!< IPv4 Socket
  uint16_t m_velocity; //!< transmitted velocity
  EventId m_sendEvent; //!< Event to send the next packet
  Ptr<TraciClient> m_client;
  std::map<std::string,std::pair<double,double>> m_vehicles_data;

  /// Callbacks for tracing the packet Tx events
  TracedCallback<Ptr<const Packet> > m_txTrace;
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

