# Stage 1: Build the Next.js application
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies
# This is crucial: uses package.json and lock file to install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy application files (including src/)
COPY . .

# Build the application
RUN npm run build

# Stage 2: Run the production application
FROM node:20-alpine AS runner

WORKDIR /app

# Copy necessary files for running
COPY --from=builder /app/package.json .
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
# You will also need next.config.ts if you have one
#COPY --from=builder next.config.ts . 

# Expose the Next.js port
EXPOSE 3000

# Start the Next.js server
CMD ["npm", "start"]