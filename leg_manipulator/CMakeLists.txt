cmake_minimum_required(VERSION 3.5)
project(leg_manipulator)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  meshes
  urdf
  textures
  DESTINATION share/${PROJECT_NAME}
)

# Also install go2_description assets so mesh paths resolve
install(DIRECTORY
  go2_description/dae
  go2_description/meshes
  DESTINATION share/${PROJECT_NAME}/go2_description
)

ament_package()
